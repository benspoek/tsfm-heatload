from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from full_year_forecasting_utils import (
    HOURLY_STEP,
    ITEM_ID,
    QUARTER_STEP,
    STEPS_PER_HOUR,
    TIMEZONE,
    UnsupportedDataCadenceError,
    add_synthetic_model_clock,
    load_residual_multiresolution_data,
    metric_values,
    sha256_file,
    to_naive_datetime,
    validate_regular_rows,
)
from utils import TABPFN_PACKAGES, make_run_id, metadata_envelope, write_metadata


DEFAULT_HEAT_PATH = Path("munich/demand/heat/heat_dh.csv")
DEFAULT_WEATHER_PATH = Path("munich/weather/munich_weather_with_solar_precipitation.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/full_year_2024/stacked")
DEFAULT_RUN_NAME = "stacked_full_year_2024_base12w_pred24h_every12h_residual7d_pred2h_every1h_temperature"

WEATHER_COLUMNS = ["temperature"]
BASE_CONTEXT_HOURS = 12 * 7 * 24
BASE_PREDICTION_HOURS = 24
BASE_FORECAST_EVERY_HOURS = 12
RESIDUAL_CONTEXT_HOURS = 7 * 24
RESIDUAL_PREDICTION_HOURS = 2
RESIDUAL_FORECAST_EVERY_HOURS = 1

BASE_HOURLY_PREDICTION_COLUMN = "base_prediction_hourly"
BASE_QUARTER_PREDICTION_COLUMN = "base_prediction_15min"
RESIDUAL_TARGET_COLUMN = "actual_residual"
RESIDUAL_PREDICTION_COLUMN = "predicted_residual"
STACKED_PREDICTION_COLUMN = "stacked_prediction"
RESIDUAL_COVARIATES = [*WEATHER_COLUMNS, BASE_QUARTER_PREDICTION_COLUMN]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full-year deterministic stacked TabPFN-TS forecast for a Munich district heating network. "
            "The base model is hourly with 12-week context and 24h horizon every 12h; "
            "the residual model is 15-minute with 7-day context and 2h horizon every hour."
        )
    )
    parser.add_argument("--heat-path", type=Path, default=DEFAULT_HEAT_PATH)
    parser.add_argument("--weather-path", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--dataset-name", default="munich")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--mode", choices=("CLIENT", "LOCAL"), default="LOCAL")
    parser.add_argument("--max-forecast-starts", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, start construction, and first windows without running predictions.",
    )
    return parser.parse_args()


def initialize_pipeline(mode: str, max_context_length: int):
    try:
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install it with: python -m pip install -r requirements.txt"
        ) from exc

    return TabPFNTSPipeline(
        tabpfn_mode=getattr(TabPFNMode, mode),
        max_context_length=max_context_length,
    )


def make_model_frame(
    data: pd.DataFrame,
    target_column: str,
    covariate_columns: list[str],
    include_target: bool,
) -> pd.DataFrame:
    out = data.copy()
    out["item_id"] = ITEM_ID
    timestamp_source = "model_timestamp" if "model_timestamp" in out.columns else "timestamp"
    out["timestamp"] = to_naive_datetime(out[timestamp_source])
    columns = ["item_id", "timestamp", *covariate_columns]
    if include_target:
        out = out.rename(columns={target_column: "target"})
        columns.insert(2, "target")
    return out[columns]


def build_window(
    data: pd.DataFrame,
    forecast_start: pd.Timestamp,
    context_hours: int,
    prediction_hours: int,
    step: pd.Timedelta,
    target_column: str,
    covariate_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context_start = forecast_start - pd.Timedelta(hours=context_hours)
    forecast_end = forecast_start + pd.Timedelta(hours=prediction_hours)
    expected_context_rows = int(pd.Timedelta(hours=context_hours) / step)
    expected_future_rows = int(pd.Timedelta(hours=prediction_hours) / step)

    context_raw = data[(data["timestamp"] >= context_start) & (data["timestamp"] < forecast_start)].copy()
    future_raw = data[(data["timestamp"] >= forecast_start) & (data["timestamp"] < forecast_end)].copy()
    if len(context_raw) != expected_context_rows:
        raise ValueError(f"Context for {forecast_start} has {len(context_raw)} rows, expected {expected_context_rows}.")
    if len(future_raw) != expected_future_rows:
        raise ValueError(f"Future for {forecast_start} has {len(future_raw)} rows, expected {expected_future_rows}.")
    required_context = [target_column, *covariate_columns]
    if context_raw[required_context].isna().any().any():
        missing_counts = context_raw[required_context].isna().sum().to_dict()
        raise ValueError(f"Missing context values for {forecast_start}: {missing_counts}")
    if future_raw[covariate_columns].isna().any().any():
        missing_counts = future_raw[covariate_columns].isna().sum().to_dict()
        raise ValueError(f"Missing future covariates for {forecast_start}: {missing_counts}")

    context_df = make_model_frame(context_raw, target_column, covariate_columns, include_target=True)
    future_df = make_model_frame(future_raw, target_column, covariate_columns, include_target=False)
    if "target" in future_df.columns:
        raise ValueError("future_df must not contain target.")
    return context_df, future_df, future_raw


def flatten_point_prediction(pred_df: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    out = pred_df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    elif "timestamp" not in out.columns:
        out = out.reset_index()
    if "timestamp" not in out.columns or "target" not in out.columns:
        raise ValueError(f"Unexpected prediction output columns: {out.columns.tolist()}")

    value_column = "target"
    for column in out.columns:
        try:
            if math.isclose(float(column), 0.5, rel_tol=0.0, abs_tol=1e-12):
                value_column = column
                break
        except (TypeError, ValueError):
            continue

    result = out[["timestamp"]].rename(columns={"timestamp": "model_timestamp"}).copy()
    result["model_timestamp"] = to_naive_datetime(result["model_timestamp"])
    result[prediction_column] = out[value_column].to_numpy(dtype=float)
    return result


def predict_point(pipeline, context_df: pd.DataFrame, future_df: pd.DataFrame, prediction_column: str) -> tuple[pd.DataFrame, float]:
    if getattr(pipeline, "max_context_length", 0) < len(context_df):
        raise ValueError(
            "TabPFN-TS would truncate context: "
            f"max_context_length={pipeline.max_context_length}, context rows={len(context_df)}."
        )
    started = time.perf_counter()
    prediction = pipeline.predict_df(context_df=context_df, future_df=future_df, quantiles=[0.5])
    prediction_seconds = time.perf_counter() - started
    return flatten_point_prediction(prediction, prediction_column), prediction_seconds


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(f"{year}-01-01 00:00:00", tz=TIMEZONE),
        pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz=TIMEZONE),
    )


def hourly_start_rows(
    hourly: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    every_hours: int,
    horizon_hours: int,
    max_count: int | None = None,
) -> pd.DataFrame:
    candidate = hourly[(hourly["timestamp"] >= start) & (hourly["timestamp"] < end_exclusive)].copy()
    candidate = candidate[candidate["timestamp"].dt.minute.eq(0)].reset_index(drop=True)
    rows = []
    expected_future_rows = int(pd.Timedelta(hours=horizon_hours) / HOURLY_STEP)
    for pos in range(0, len(candidate), every_hours):
        forecast_start = candidate.loc[pos, "timestamp"]
        forecast_end = forecast_start + pd.Timedelta(hours=horizon_hours)
        future_rows = hourly[(hourly["timestamp"] >= forecast_start) & (hourly["timestamp"] < forecast_end)]
        if len(future_rows) != expected_future_rows:
            continue
        rows.append({"forecast_start": forecast_start})
        if max_count is not None and len(rows) >= max_count:
            break
    return pd.DataFrame(rows)


def residual_start_rows(
    quarter: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    every_hours: int,
    horizon_hours: int,
    max_count: int | None = None,
) -> pd.DataFrame:
    candidate = quarter[
        (quarter["timestamp"] >= start)
        & (quarter["timestamp"] < end_exclusive)
        & quarter["timestamp"].dt.minute.eq(0)
    ].reset_index(drop=True)
    rows = []
    expected_future_rows = int(pd.Timedelta(hours=horizon_hours) / QUARTER_STEP)
    for pos in range(0, len(candidate), every_hours):
        forecast_start = candidate.loc[pos, "timestamp"]
        forecast_end = forecast_start + pd.Timedelta(hours=horizon_hours)
        if forecast_end > end_exclusive:
            continue
        future_rows = quarter[(quarter["timestamp"] >= forecast_start) & (quarter["timestamp"] < forecast_end)]
        if len(future_rows) != expected_future_rows:
            continue
        rows.append({"forecast_start": forecast_start})
        if max_count is not None and len(rows) >= max_count:
            break
    return pd.DataFrame(rows)


def run_base_forecasts(hourly: pd.DataFrame, starts: pd.DataFrame, mode: str) -> pd.DataFrame:
    context_rows = BASE_CONTEXT_HOURS
    pipeline = initialize_pipeline(mode, max_context_length=context_rows)
    rows = []
    for idx, start_row in starts.iterrows():
        forecast_start = start_row["forecast_start"]
        print(f"[base {idx + 1}/{len(starts)}] {forecast_start}")
        context_df, future_df, future_raw = build_window(
            data=hourly,
            forecast_start=forecast_start,
            context_hours=BASE_CONTEXT_HOURS,
            prediction_hours=BASE_PREDICTION_HOURS,
            step=HOURLY_STEP,
            target_column="heat",
            covariate_columns=WEATHER_COLUMNS,
        )
        prediction, prediction_seconds = predict_point(
            pipeline,
            context_df=context_df,
            future_df=future_df,
            prediction_column=BASE_HOURLY_PREDICTION_COLUMN,
        )
        out = future_raw.merge(prediction, on="model_timestamp", how="left", validate="one_to_one")
        if out[BASE_HOURLY_PREDICTION_COLUMN].isna().any():
            raise ValueError(f"Missing base predictions for {forecast_start}.")
        out.insert(0, "model", "hourly_base")
        out.insert(1, "forecast_start", forecast_start)
        out.insert(2, "horizon_step", np.arange(1, len(out) + 1))
        out.insert(3, "horizon_minutes", out["horizon_step"] * 60)
        out.insert(4, "context_rows", len(context_df))
        out.insert(5, "prediction_seconds", prediction_seconds)
        rows.append(out.rename(columns={"heat": "actual_heat_hourly"}))
    return pd.concat(rows, ignore_index=True)


def collapse_latest(raw: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    latest = (
        raw.sort_values(["timestamp", "forecast_start"])
        .groupby("timestamp", as_index=False)
        .tail(1)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return latest[["timestamp", "forecast_start", *value_columns]].copy()


def available_base(raw_base: pd.DataFrame, available_at: pd.Timestamp) -> pd.DataFrame:
    available = raw_base[raw_base["forecast_start"] <= available_at]
    if available.empty:
        raise ValueError(f"No hourly base forecasts are available at {available_at}.")
    return collapse_latest(available, [BASE_HOURLY_PREDICTION_COLUMN])


def interpolate_hourly_to_quarter(
    hourly_predictions: pd.DataFrame,
    quarter_timestamps: pd.Series,
    source_column: str,
    target_column: str,
) -> pd.DataFrame:
    base = hourly_predictions.set_index("timestamp")[source_column].sort_index()
    target_index = pd.DatetimeIndex(quarter_timestamps.sort_values().unique())
    combined_index = base.index.union(target_index).sort_values()
    interpolated = base.reindex(combined_index).interpolate(method="time").ffill().bfill().reindex(target_index)
    out = pd.DataFrame({"timestamp": target_index, target_column: interpolated.to_numpy()})
    if out[target_column].isna().any():
        missing = out.loc[out[target_column].isna(), "timestamp"].head().tolist()
        raise ValueError(f"Hourly-to-quarter interpolation produced missing values, examples: {missing}")
    return out


def build_residual_window(
    quarter: pd.DataFrame,
    raw_base: pd.DataFrame,
    forecast_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context_start = forecast_start - pd.Timedelta(hours=RESIDUAL_CONTEXT_HOURS)
    forecast_end = forecast_start + pd.Timedelta(hours=RESIDUAL_PREDICTION_HOURS)
    context_raw = quarter[(quarter["timestamp"] >= context_start) & (quarter["timestamp"] < forecast_start)].copy()
    future_raw = quarter[(quarter["timestamp"] >= forecast_start) & (quarter["timestamp"] < forecast_end)].copy()
    expected_context_rows = RESIDUAL_CONTEXT_HOURS * STEPS_PER_HOUR
    expected_future_rows = RESIDUAL_PREDICTION_HOURS * STEPS_PER_HOUR
    if len(context_raw) != expected_context_rows:
        raise ValueError(f"Residual context for {forecast_start} has {len(context_raw)} rows, expected {expected_context_rows}.")
    if len(future_raw) != expected_future_rows:
        raise ValueError(f"Residual future for {forecast_start} has {len(future_raw)} rows, expected {expected_future_rows}.")

    base_at_start = available_base(raw_base, available_at=forecast_start)
    context_base = interpolate_hourly_to_quarter(
        base_at_start,
        context_raw["timestamp"],
        BASE_HOURLY_PREDICTION_COLUMN,
        BASE_QUARTER_PREDICTION_COLUMN,
    )
    future_base = interpolate_hourly_to_quarter(
        base_at_start,
        future_raw["timestamp"],
        BASE_HOURLY_PREDICTION_COLUMN,
        BASE_QUARTER_PREDICTION_COLUMN,
    )
    context_raw = context_raw.merge(context_base, on="timestamp", how="left", validate="one_to_one")
    future_raw = future_raw.merge(future_base, on="timestamp", how="left", validate="one_to_one")
    context_raw[RESIDUAL_TARGET_COLUMN] = context_raw["heat"] - context_raw[BASE_QUARTER_PREDICTION_COLUMN]
    future_raw[RESIDUAL_TARGET_COLUMN] = future_raw["heat"] - future_raw[BASE_QUARTER_PREDICTION_COLUMN]

    context_df = make_model_frame(context_raw, RESIDUAL_TARGET_COLUMN, RESIDUAL_COVARIATES, include_target=True)
    future_df = make_model_frame(future_raw, RESIDUAL_TARGET_COLUMN, RESIDUAL_COVARIATES, include_target=False)
    if "target" in future_df.columns:
        raise ValueError("Residual future_df must not contain target.")
    return context_df, future_df, future_raw


def run_residual_forecasts(quarter: pd.DataFrame, raw_base: pd.DataFrame, starts: pd.DataFrame, mode: str) -> pd.DataFrame:
    pipeline = initialize_pipeline(mode, max_context_length=RESIDUAL_CONTEXT_HOURS * STEPS_PER_HOUR)
    rows = []
    for idx, start_row in starts.iterrows():
        forecast_start = start_row["forecast_start"]
        print(f"[residual {idx + 1}/{len(starts)}] {forecast_start}")
        context_df, future_df, future_raw = build_residual_window(quarter, raw_base, forecast_start)
        prediction, prediction_seconds = predict_point(
            pipeline,
            context_df=context_df,
            future_df=future_df,
            prediction_column=RESIDUAL_PREDICTION_COLUMN,
        )
        out = future_raw.merge(prediction, on="model_timestamp", how="left", validate="one_to_one")
        if out[RESIDUAL_PREDICTION_COLUMN].isna().any():
            raise ValueError(f"Missing residual predictions for {forecast_start}.")
        out.insert(0, "model", "stacked_residual")
        out.insert(1, "forecast_start", forecast_start)
        out.insert(2, "horizon_step", np.arange(1, len(out) + 1))
        out.insert(3, "horizon_minutes", out["horizon_step"] * 15)
        out.insert(4, "context_rows", len(context_df))
        out.insert(5, "prediction_seconds", prediction_seconds)
        out[STACKED_PREDICTION_COLUMN] = out[BASE_QUARTER_PREDICTION_COLUMN] + out[RESIDUAL_PREDICTION_COLUMN]
        rows.append(out.rename(columns={"heat": "actual_heat"}))
    return pd.concat(rows, ignore_index=True)


def build_final_predictions(
    quarter: pd.DataFrame,
    raw_residual: pd.DataFrame,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> pd.DataFrame:
    actual = quarter[(quarter["timestamp"] >= eval_start) & (quarter["timestamp"] < eval_end)][
        ["timestamp", "heat", *WEATHER_COLUMNS]
    ].rename(columns={"heat": "actual_heat"})
    stacked = collapse_latest(
        raw_residual,
        [BASE_QUARTER_PREDICTION_COLUMN, RESIDUAL_PREDICTION_COLUMN, STACKED_PREDICTION_COLUMN],
    ).rename(columns={"forecast_start": "residual_forecast_start"})
    final = actual.merge(stacked, on="timestamp", how="left", validate="one_to_one")
    required = [BASE_QUARTER_PREDICTION_COLUMN, RESIDUAL_PREDICTION_COLUMN, STACKED_PREDICTION_COLUMN]
    if final[required].isna().any().any():
        missing_counts = final[required].isna().sum().to_dict()
        raise ValueError(f"Missing final predictions: {missing_counts}")
    return final


def calculate_summary(raw_predictions: pd.DataFrame, raw_base: pd.DataFrame, raw_residual: pd.DataFrame) -> pd.DataFrame:
    model_columns = {
        "hourly_base_interpolated": BASE_QUARTER_PREDICTION_COLUMN,
        "stacked": STACKED_PREDICTION_COLUMN,
    }
    rows = []
    for model, prediction_column in model_columns.items():
        row: dict[str, object] = {
            "model": model,
            "metric_scope": "all",
            "n_rows": int(len(raw_predictions)),
            "evaluation_resolution": "15min",
        }
        row.update(metric_values(raw_predictions["actual_heat"], raw_predictions[prediction_column]))
        timing_source = raw_base if model == "hourly_base_interpolated" else raw_residual
        per_start = timing_source.groupby("forecast_start", sort=False)["prediction_seconds"].first()
        row["n_forecast_starts"] = int(len(per_start))
        row["prediction_seconds_total"] = float(per_start.sum())
        row["prediction_seconds_mean_per_forecast_start"] = float(per_start.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_per_start_metrics(raw_residual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for forecast_start, group in raw_residual.groupby("forecast_start", sort=True):
        row: dict[str, object] = {
            "model": "stacked",
            "forecast_start": forecast_start.isoformat(),
            "n_rows": int(len(group)),
            "evaluation_resolution": "15min",
            "prediction_seconds": float(group["prediction_seconds"].iloc[0]),
        }
        row.update(metric_values(group["actual_heat"], group[STACKED_PREDICTION_COLUMN]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_metadata(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    n_base_starts: int,
    n_residual_starts: int,
    total_seconds: float,
) -> dict[str, object]:
    doc = metadata_envelope(
        run_id=run_id,
        run_name=args.run_name,
        run_dir=run_dir,
        script_path=__file__,
        packages=TABPFN_PACKAGES,
    )
    doc.update({
        "inputs": {
            "dataset_name": args.dataset_name,
            "heat_path": str(args.heat_path),
            "heat_sha256": sha256_file(args.heat_path),
            "weather_path": str(args.weather_path),
            "weather_sha256": sha256_file(args.weather_path),
            "weather_columns": WEATHER_COLUMNS,
        },
        "evaluation": {
            "year": args.year,
            "resolution": "15min",
            "includes_signal_outage_periods": True,
            "future_covariates_exclude_target": True,
            "timestamp_axis": (
                "Base and residual TabPFN-TS calls receive synthetic continuous model clocks. "
                "Real Europe/Berlin timestamps are retained for scheduling, interpolation, and outputs."
            ),
        },
        "stacked_predictor": {
            "base": {
                "resolution": "hourly",
                "context_hours": BASE_CONTEXT_HOURS,
                "prediction_hours": BASE_PREDICTION_HOURS,
                "forecast_every_hours": BASE_FORECAST_EVERY_HOURS,
                "covariates": WEATHER_COLUMNS,
                "n_forecast_starts": n_base_starts,
            },
            "residual": {
                "resolution": "15min",
                "context_hours": RESIDUAL_CONTEXT_HOURS,
                "prediction_hours": RESIDUAL_PREDICTION_HOURS,
                "forecast_every_hours": RESIDUAL_FORECAST_EVERY_HOURS,
                "covariates": RESIDUAL_COVARIATES,
                "target": "actual heat - available base prediction",
                "n_forecast_starts": n_residual_starts,
            },
            "final_prediction": "base_prediction_15min + predicted_residual",
            "overlap_policy": "latest residual forecast wins",
        },
        "total_seconds": total_seconds,
    })
    return doc


def validate_forecast_windows(
    quarter: pd.DataFrame,
    hourly: pd.DataFrame,
    base_starts: pd.DataFrame,
    residual_starts: pd.DataFrame,
) -> tuple[int, int]:
    for forecast_start in base_starts["forecast_start"]:
        _, future, _ = build_window(
            hourly,
            forecast_start=forecast_start,
            context_hours=BASE_CONTEXT_HOURS,
            prediction_hours=BASE_PREDICTION_HOURS,
            step=HOURLY_STEP,
            target_column="heat",
            covariate_columns=WEATHER_COLUMNS,
        )
        if "target" in future.columns:
            raise ValueError("Base future_df must not contain target.")

    for forecast_start in residual_starts["forecast_start"]:
        fake_base = hourly[
            (hourly["timestamp"] >= forecast_start - pd.Timedelta(hours=RESIDUAL_CONTEXT_HOURS + 1))
            & (hourly["timestamp"] <= forecast_start + pd.Timedelta(hours=RESIDUAL_PREDICTION_HOURS))
        ][["timestamp", "heat"]].rename(columns={"heat": BASE_HOURLY_PREDICTION_COLUMN})
        fake_base.insert(0, "forecast_start", forecast_start - pd.Timedelta(hours=BASE_FORECAST_EVERY_HOURS))
        _, future, _ = build_residual_window(quarter, fake_base, forecast_start)
        if "target" in future.columns:
            raise ValueError("Residual future_df must not contain target.")
    return len(base_starts), len(residual_starts)


def main() -> None:
    args = parse_args()
    eval_start, eval_end = year_bounds(args.year)
    try:
        quarter, hourly = load_residual_multiresolution_data(
            args.heat_path,
            args.weather_path,
            WEATHER_COLUMNS,
        )
    except UnsupportedDataCadenceError as exc:
        raise SystemExit(str(exc)) from None

    validation_start = eval_start - pd.Timedelta(hours=BASE_CONTEXT_HOURS + RESIDUAL_CONTEXT_HOURS + BASE_PREDICTION_HOURS)
    validate_regular_rows(
        quarter[(quarter["timestamp"] >= validation_start) & (quarter["timestamp"] < eval_end)],
        QUARTER_STEP,
        "quarter-hour merged data",
    )
    validate_regular_rows(
        hourly[(hourly["timestamp"] >= validation_start) & (hourly["timestamp"] < eval_end)],
        HOURLY_STEP,
        "hourly merged data",
    )
    quarter = add_synthetic_model_clock(quarter, QUARTER_STEP)
    hourly = add_synthetic_model_clock(hourly, HOURLY_STEP)

    base_start_min = eval_start - pd.Timedelta(hours=RESIDUAL_CONTEXT_HOURS + BASE_PREDICTION_HOURS)
    base_starts = hourly_start_rows(
        hourly,
        start=base_start_min,
        end_exclusive=eval_end,
        every_hours=BASE_FORECAST_EVERY_HOURS,
        horizon_hours=BASE_PREDICTION_HOURS,
        max_count=None if args.max_forecast_starts is None else args.max_forecast_starts + 20,
    )
    residual_starts = residual_start_rows(
        quarter,
        start=eval_start,
        end_exclusive=eval_end,
        every_hours=RESIDUAL_FORECAST_EVERY_HOURS,
        horizon_hours=RESIDUAL_PREDICTION_HOURS,
        max_count=args.max_forecast_starts,
    )
    if base_starts.empty or residual_starts.empty:
        raise ValueError("No base or residual forecast starts were constructed.")

    print(f"Dataset: {args.dataset_name}")
    print(f"Base starts: {len(base_starts):,}")
    print(f"Residual starts: {len(residual_starts):,}")
    print(f"Evaluation period: {eval_start} <= timestamp < {eval_end}")

    if args.dry_run:
        n_base, n_residual = validate_forecast_windows(quarter, hourly, base_starts, residual_starts)
        print(
            f"Dry run complete. Validated {n_base:,} base windows and "
            f"{n_residual:,} residual windows; no predictions or outputs written."
        )
        return

    total_start = time.perf_counter()
    run_id = make_run_id(args.run_id, args.run_name)
    run_dir = args.output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    raw_base = run_base_forecasts(hourly, base_starts, args.mode)
    if args.max_forecast_starts is not None:
        # Keep enough base warm-up starts, but evaluate only the requested residual starts.
        raw_base = raw_base.copy()

    raw_residual = run_residual_forecasts(quarter, raw_base, residual_starts, args.mode)
    eval_cutoff = residual_starts["forecast_start"].max() + pd.Timedelta(hours=RESIDUAL_PREDICTION_HOURS)
    effective_eval_end = min(eval_end, eval_cutoff)
    raw_predictions = build_final_predictions(
        quarter,
        raw_residual,
        eval_start=eval_start,
        eval_end=effective_eval_end,
    )

    summary = calculate_summary(raw_predictions, raw_base, raw_residual)
    per_start = calculate_per_start_metrics(raw_residual)

    raw_base.to_csv(run_dir / "raw_base_forecasts.csv", index=False)
    raw_residual.to_csv(run_dir / "raw_residual_forecasts.csv", index=False)
    raw_predictions.to_csv(run_dir / "raw_predictions.csv", index=False)
    per_start.to_csv(run_dir / "metrics_per_forecast_start.csv", index=False)
    summary.to_csv(run_dir / "metrics_summary.csv", index=False)

    total_seconds = time.perf_counter() - total_start
    write_metadata(
        run_dir / "run_metadata.json",
        build_metadata(
            args=args,
            run_id=run_id,
            run_dir=run_dir,
            n_base_starts=len(base_starts),
            n_residual_starts=len(residual_starts),
            total_seconds=total_seconds,
        ),
    )

    print(f"Saved run directory: {run_dir}")
    print(f"Saved raw base forecasts: {run_dir / 'raw_base_forecasts.csv'}")
    print(f"Saved raw residual forecasts: {run_dir / 'raw_residual_forecasts.csv'}")
    print(f"Saved raw predictions: {run_dir / 'raw_predictions.csv'}")
    print(f"Saved per-start metrics: {run_dir / 'metrics_per_forecast_start.csv'}")
    print(f"Saved summary metrics: {run_dir / 'metrics_summary.csv'}")
    print(f"Saved metadata: {run_dir / 'run_metadata.json'}")
    print(f"Total seconds: {total_seconds:.2f}")


if __name__ == "__main__":
    main()
