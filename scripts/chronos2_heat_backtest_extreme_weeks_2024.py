from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from full_year_forecasting_utils import (
    DEFAULT_SELECTED_WEEKS_PATH,
    HOURLY_STEP,
    ITEM_ID,
    QUARTER_STEP,
    TIMEZONE,
    UnsupportedDataCadenceError,
    add_synthetic_model_clock,
    forecast_starts_for_selected_weeks,
    load_merged_data,
    load_selected_weeks,
    sha256_file,
    validate_complete_year,
)
from utils import (
    CHRONOS_PACKAGES,
    make_run_id,
    metadata_envelope,
    parse_weather_columns,
    write_metadata,
)


DEFAULT_HEAT_PATH = Path("flensburg/demand/heat/heat_dh.csv")
DEFAULT_WEATHER_PATH = Path("flensburg/weather/flensburg_weather_temperature.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/extreme_weeks_2024")
DEFAULT_MODEL_PATH = "amazon/chronos-2"
DEFAULT_MAX_CONTEXT_STEPS = 8192
MODEL_NAME = "Chronos-2"
RESOLUTION_TO_STEP = {"quarter": QUARTER_STEP, "hourly": HOURLY_STEP}
STEPS_PER_HOUR_BY_RESOLUTION = {"quarter": 4, "hourly": 1}
DEFAULT_QUANTILES = [
    0.01,
    0.025,
    0.05,
    0.10,
    0.1587,
    0.20,
    0.25,
    0.50,
    0.75,
    0.80,
    0.8413,
    0.90,
    0.95,
    0.975,
    0.99,
]
INTERVAL_DEFINITIONS = {
    "50": ("q25", "q75"),
    "68": ("q15_87", "q84_13"),
    "80": ("q10", "q90"),
    "90": ("q05", "q95"),
    "95": ("q02_5", "q97_5"),
    "98": ("q01", "q99"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run direct Chronos-2 24h heat-demand backtests on the three "
            "signal-error-free 2024 extreme weeks."
        )
    )
    parser.add_argument("--heat-path", type=Path, default=DEFAULT_HEAT_PATH)
    parser.add_argument("--weather-path", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--selected-weeks-path", type=Path, default=DEFAULT_SELECTED_WEEKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resolution", choices=tuple(RESOLUTION_TO_STEP), required=True)
    parser.add_argument("--prediction-hours", type=int, default=24)
    parser.add_argument("--context-hours", type=int, required=True)
    parser.add_argument("--weather-columns", default="temperature")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--max-context-steps", type=int, default=DEFAULT_MAX_CONTEXT_STEPS)
    parser.add_argument("--max-forecast-starts", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, selected weeks, forecast starts, and windows without loading Chronos-2.",
    )
    return parser.parse_args()


def prediction_steps(prediction_hours: int, resolution: str) -> int:
    if prediction_hours <= 0:
        raise ValueError(f"prediction-hours must be positive, got {prediction_hours}.")
    return prediction_hours * STEPS_PER_HOUR_BY_RESOLUTION[resolution]


def requested_context_steps(context_hours: int, resolution: str) -> int:
    if context_hours <= 0:
        raise ValueError(f"context-hours must be positive, got {context_hours}.")
    return context_hours * STEPS_PER_HOUR_BY_RESOLUTION[resolution]


def effective_context_steps(context_hours: int, resolution: str, max_context_steps: int) -> int:
    if max_context_steps <= 0:
        raise ValueError(f"max-context-steps must be positive, got {max_context_steps}.")
    return min(requested_context_steps(context_hours, resolution), max_context_steps)


def quantile_column_name(value: float) -> str:
    percent = value * 100
    text = f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "_")
    if percent < 10 and not text.startswith("0"):
        text = f"0{text}"
    return f"q{text}"


def find_column_by_numeric_value(columns: pd.Index, value: float) -> object | None:
    for column in columns:
        try:
            if math.isclose(float(column), value, rel_tol=0.0, abs_tol=1e-12):
                return column
        except (TypeError, ValueError):
            continue
    return None


def make_chronos_frame(
    data: pd.DataFrame,
    weather_columns: list[str],
    include_target: bool,
) -> pd.DataFrame:
    out = data.copy()
    out["item_id"] = ITEM_ID
    out["timestamp"] = pd.to_datetime(out["model_timestamp"]).dt.tz_localize(None)
    columns = ["item_id", "timestamp", *weather_columns]
    if include_target:
        out = out.rename(columns={"heat": "target"})
        columns.insert(2, "target")
    return out[columns]


def build_window(
    data: pd.DataFrame,
    forecast_start: pd.Timestamp,
    args: argparse.Namespace,
    weather_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    matches = data.index[data["timestamp"].eq(forecast_start)].tolist()
    if len(matches) != 1:
        raise ValueError(f"Forecast start {forecast_start} not found exactly once.")

    start_pos = matches[0]
    req_steps = requested_context_steps(args.context_hours, args.resolution)
    eff_steps = effective_context_steps(args.context_hours, args.resolution, args.max_context_steps)
    horizon_steps = prediction_steps(args.prediction_hours, args.resolution)
    context_start_pos = start_pos - eff_steps
    horizon_end_pos = start_pos + horizon_steps

    if context_start_pos < 0:
        raise ValueError(
            f"Not enough effective context before {forecast_start}. "
            f"Need {eff_steps} rows, have {start_pos}."
        )
    if horizon_end_pos > len(data):
        raise ValueError(f"Not enough future rows after {forecast_start}.")

    context_raw = data.iloc[context_start_pos:start_pos].copy()
    future_raw = data.iloc[start_pos:horizon_end_pos].copy()
    required_context = ["heat", *weather_columns]
    if context_raw[required_context].isna().any().any():
        missing_counts = context_raw[required_context].isna().sum().to_dict()
        raise ValueError(f"Missing context values at {forecast_start}: {missing_counts}")
    if future_raw[weather_columns].isna().any().any():
        missing_counts = future_raw[weather_columns].isna().sum().to_dict()
        raise ValueError(f"Missing future covariates at {forecast_start}: {missing_counts}")

    context_df = make_chronos_frame(context_raw, weather_columns, include_target=True)
    future_df = make_chronos_frame(future_raw, weather_columns, include_target=False)
    if "target" in future_df.columns:
        raise ValueError("future_df must not contain target.")

    actual = future_raw[["timestamp", "model_timestamp", "heat", *weather_columns]].rename(
        columns={"heat": "actual_heat"}
    )
    actual["model_timestamp"] = pd.to_datetime(actual["model_timestamp"]).dt.tz_localize(None)
    info = {
        "requested_context_steps": req_steps,
        "effective_context_steps": eff_steps,
        "requested_context_hours": args.context_hours,
        "effective_context_hours": eff_steps / STEPS_PER_HOUR_BY_RESOLUTION[args.resolution],
        "context_capped": req_steps > eff_steps,
        "context_start": context_raw["timestamp"].iloc[0],
        "requested_context_start": forecast_start - pd.Timedelta(hours=args.context_hours),
    }
    return context_df, future_df, actual, info


def initialize_pipeline(args: argparse.Namespace):
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install Chronos-2 direct inference support with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    return Chronos2Pipeline.from_pretrained(args.model_path, device_map=args.device_map)


def flatten_chronos_predictions(pred_df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    out = pred_df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    elif "timestamp" not in out.columns:
        out = out.reset_index()

    if "timestamp" not in out.columns:
        raise ValueError(f"Prediction output has no timestamp column: {out.columns.tolist()}")

    out["model_timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_localize(None)
    result = out[["model_timestamp"]].copy()

    if "target" in out.columns:
        result["predicted_heat"] = pd.to_numeric(out["target"], errors="raise")
    elif "mean" in out.columns:
        result["predicted_heat"] = pd.to_numeric(out["mean"], errors="raise")
    else:
        median_column = find_column_by_numeric_value(out.columns, 0.5)
        if median_column is None:
            raise ValueError(
                "Prediction output has no point forecast column. Expected one of "
                f"'target', 'mean', or quantile 0.5. Columns: {out.columns.tolist()}"
            )
        result["predicted_heat"] = pd.to_numeric(out[median_column], errors="raise")

    for quantile in quantiles:
        target_column = quantile_column_name(quantile)
        source_column = find_column_by_numeric_value(out.columns, quantile)
        if source_column is not None:
            result[target_column] = pd.to_numeric(out[source_column], errors="raise")
        elif math.isclose(quantile, 0.5, rel_tol=0.0, abs_tol=1e-12):
            result[target_column] = result["predicted_heat"]
        else:
            result[target_column] = np.nan

    if "q50" in result.columns:
        result["predicted_heat"] = result["q50"]
    return result


def build_forecast_output(
    pred_df: pd.DataFrame,
    actual: pd.DataFrame,
    step: pd.Timedelta,
    quantiles: list[float],
) -> pd.DataFrame:
    pred = flatten_chronos_predictions(pred_df, quantiles)
    out = actual.merge(pred, on="model_timestamp", how="left", validate="one_to_one")
    if out["predicted_heat"].isna().any():
        missing = out.loc[out["predicted_heat"].isna(), "model_timestamp"].head().tolist()
        raise ValueError(f"Some forecast timestamps did not receive predictions: {missing}")
    out.insert(0, "horizon_step", np.arange(1, len(out) + 1))
    out.insert(1, "horizon_minutes", out["horizon_step"] * int(step.total_seconds() / 60))
    out["error"] = out["predicted_heat"] - out["actual_heat"]
    out["absolute_error"] = out["error"].abs()
    denominator = out["actual_heat"].abs() + out["predicted_heat"].abs()
    out["sape"] = np.where(denominator > 0, 2 * out["absolute_error"] / denominator, np.nan)
    return out


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    error = actual - predicted
    return float(np.nanmean(np.maximum(quantile * error, (quantile - 1) * error)))


def add_interval_metrics(metrics: dict[str, object], forecast: pd.DataFrame) -> None:
    actual = forecast["actual_heat"]
    for label, (lower_column, upper_column) in INTERVAL_DEFINITIONS.items():
        if lower_column not in forecast.columns or upper_column not in forecast.columns:
            continue
        lower = forecast[lower_column]
        upper = forecast[upper_column]
        if lower.isna().all() or upper.isna().all():
            continue
        inside = (actual >= lower) & (actual <= upper)
        width = upper - lower
        metrics[f"coverage_{label}_percent"] = float(inside.mean() * 100)
        metrics[f"mean_width_{label}"] = float(width.mean())


def metric_values(forecast: pd.DataFrame) -> dict[str, float]:
    error = forecast["error"].to_numpy(dtype=float)
    actual = forecast["actual_heat"].to_numpy(dtype=float)
    absolute_error = np.abs(error)
    rmse = math.sqrt(float(np.mean(error**2)))
    actual_mean = float(np.mean(actual))
    total_sum_of_squares = float(np.sum((actual - actual_mean) ** 2))
    residual_sum_of_squares = float(np.sum(error**2))
    return {
        "MAE": float(np.mean(absolute_error)),
        "RMSE": rmse,
        "R2": float(1 - residual_sum_of_squares / total_sum_of_squares) if total_sum_of_squares > 0 else np.nan,
        "CVRMSE_percent": float(rmse / actual_mean * 100) if actual_mean != 0 else np.nan,
        "sMAPE_percent": float(np.nanmean(forecast["sape"]) * 100),
        "mean_error": float(np.mean(error)),
        "median_absolute_error": float(np.median(absolute_error)),
        "max_absolute_error": float(np.max(absolute_error)),
    }


def calculate_forecast_metrics(
    forecast: pd.DataFrame,
    args: argparse.Namespace,
    selection: str,
    week_id: str,
    forecast_start: pd.Timestamp,
    data_loading_seconds: float,
    prediction_seconds: float,
    context_info: dict[str, object],
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "model": MODEL_NAME,
        "selection": selection,
        "week_id": week_id,
        "forecast_start": forecast_start.tz_localize(None).isoformat(),
        "resolution": args.resolution,
        "context_hours": args.context_hours,
        "context_days_equivalent": args.context_hours / 24,
        "requested_context_hours": context_info["requested_context_hours"],
        "effective_context_hours": context_info["effective_context_hours"],
        "requested_context_steps": context_info["requested_context_steps"],
        "effective_context_steps": context_info["effective_context_steps"],
        "context_capped": context_info["context_capped"],
        "prediction_hours": args.prediction_hours,
        "prediction_steps": prediction_steps(args.prediction_hours, args.resolution),
        "n_context_rows": context_info["effective_context_steps"],
        "n_forecast_rows": len(forecast),
        "data_loading_seconds": data_loading_seconds,
        "prediction_seconds": prediction_seconds,
        "total_seconds": data_loading_seconds + prediction_seconds,
    }
    metrics.update(metric_values(forecast))
    add_interval_metrics(metrics, forecast)

    actual = forecast["actual_heat"].to_numpy(dtype=float)
    for quantile in DEFAULT_QUANTILES:
        column = quantile_column_name(quantile)
        if column in forecast.columns and forecast[column].notna().any():
            metrics[f"pinball_loss_{column}"] = pinball_loss(actual, forecast[column].to_numpy(dtype=float), quantile)
    return metrics


def add_run_timing_metrics(metrics: dict[str, object], group: pd.DataFrame) -> None:
    per_forecast_start = group.groupby("forecast_start", sort=False)["prediction_seconds"].first()
    prediction_times = per_forecast_start.to_numpy(dtype=float)
    first_prediction = float(prediction_times[0])
    mean_all = float(np.mean(prediction_times))
    if len(prediction_times) > 1:
        mean_excluding_first = float(np.mean(prediction_times[1:]))
        estimated_setup_time = first_prediction - mean_excluding_first
    else:
        mean_excluding_first = np.nan
        estimated_setup_time = np.nan
    metrics.update(
        {
            "prediction_seconds_first": first_prediction,
            "prediction_seconds_mean_all": mean_all,
            "prediction_seconds_mean_excluding_first": mean_excluding_first,
            "estimated_setup_time": estimated_setup_time,
        }
    )
    if "forecast_loop_seconds" in group.columns:
        per_forecast_loop = group.groupby("forecast_start", sort=False)["forecast_loop_seconds"].first()
        loop_times = per_forecast_loop.to_numpy(dtype=float)
        metrics.update(
            {
                "forecast_loop_seconds_first": float(loop_times[0]),
                "forecast_loop_seconds_mean_all": float(np.mean(loop_times)),
            }
        )


def calculate_group_metrics(scope: str, group: pd.DataFrame) -> dict[str, object]:
    metrics: dict[str, object] = {
        "model": MODEL_NAME,
        "metric_scope": scope,
        "n_forecast_starts": int(group["forecast_start"].nunique()),
        "n_rows": int(len(group)),
        "resolution": ",".join(sorted(group["resolution"].astype(str).unique())),
        "context_hours": ",".join(map(str, sorted(group["context_hours"].unique()))),
        "requested_context_hours": ",".join(map(str, sorted(group["requested_context_hours"].unique()))),
        "effective_context_hours": ",".join(map(str, sorted(group["effective_context_hours"].unique()))),
        "requested_context_steps": ",".join(map(str, sorted(group["requested_context_steps"].unique()))),
        "effective_context_steps": ",".join(map(str, sorted(group["effective_context_steps"].unique()))),
        "context_capped": bool(group["context_capped"].any()),
        "prediction_hours": ",".join(map(str, sorted(group["prediction_hours"].unique()))),
        "prediction_steps": ",".join(map(str, sorted(group["prediction_steps"].unique()))),
        "prediction_seconds_total": float(group.groupby("forecast_start", sort=False)["prediction_seconds"].first().sum()),
        "prediction_seconds_mean_per_forecast_start": float(
            group.groupby("forecast_start", sort=False)["prediction_seconds"].first().mean()
        ),
    }
    if "forecast_loop_seconds" in group.columns:
        per_forecast_loop = group.groupby("forecast_start", sort=False)["forecast_loop_seconds"].first()
        metrics["forecast_loop_seconds_total"] = float(per_forecast_loop.sum())
        metrics["forecast_loop_seconds_mean_per_forecast_start"] = float(per_forecast_loop.mean())
    metrics.update(metric_values(group))
    add_interval_metrics(metrics, group)
    if scope == "all":
        add_run_timing_metrics(metrics, group)
    return metrics


def calculate_summary_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    rows = [calculate_group_metrics("all", raw)]
    rows.extend(
        calculate_group_metrics(f"selection:{selection}", group)
        for selection, group in raw.groupby("selection", sort=True)
    )
    rows.extend(
        calculate_group_metrics(f"week:{week_id}", group)
        for week_id, group in raw.groupby("week_id", sort=True)
    )
    return pd.DataFrame(rows)


def build_metadata(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    selected_weeks: pd.DataFrame,
    weather_columns: list[str],
    total_seconds: float,
) -> dict[str, object]:
    metadata_payload = metadata_envelope(
        run_id=run_id,
        run_name=args.run_name,
        run_dir=run_dir,
        script_path=__file__,
        packages=CHRONOS_PACKAGES,
    )
    metadata_payload.update({
        "args": {
            "resolution": args.resolution,
            "context_hours": args.context_hours,
            "requested_context_steps": requested_context_steps(args.context_hours, args.resolution),
            "max_context_steps": args.max_context_steps,
            "effective_context_steps": effective_context_steps(args.context_hours, args.resolution, args.max_context_steps),
            "effective_context_hours": effective_context_steps(
                args.context_hours,
                args.resolution,
                args.max_context_steps,
            )
            / STEPS_PER_HOUR_BY_RESOLUTION[args.resolution],
            "context_capped": requested_context_steps(args.context_hours, args.resolution) > args.max_context_steps,
            "prediction_hours": args.prediction_hours,
            "prediction_steps": prediction_steps(args.prediction_hours, args.resolution),
            "model_path": args.model_path,
            "device_map": args.device_map,
            "weather_columns": weather_columns,
            "output_dir": str(args.output_dir),
        },
        "selected_forecast_weeks": selected_weeks.to_dict(orient="records"),
        "data": {
            "heat_path": str(args.heat_path),
            "heat_sha256": sha256_file(args.heat_path),
            "weather_path": str(args.weather_path),
            "weather_sha256": sha256_file(args.weather_path),
            "selected_weeks_path": str(args.selected_weeks_path),
            "selected_weeks_sha256": sha256_file(args.selected_weeks_path),
        },
        "model": {
            "name": MODEL_NAME,
            "implementation": "chronos.Chronos2Pipeline",
            "model_path": args.model_path,
            "direct_chronos": True,
            "autogluon_used": False,
        },
        "time_handling": {
            "timezone": TIMEZONE,
            "model_timestamp": "synthetic continuous naive timestamp axis",
            "output_timestamp": "original Europe/Berlin timestamp retained in raw_predictions.csv",
        },
        "forecast_windowing": {
            "mode": "non_overlapping_within_selected_week",
            "window_hours": args.prediction_hours,
            "step_hours": args.prediction_hours,
            "week_duration_hours": 168,
            "incomplete_final_windows": "discarded",
        },
        "quantiles": DEFAULT_QUANTILES,
        "interval_definitions": INTERVAL_DEFINITIONS,
        "total_seconds": total_seconds,
    })
    return metadata_payload


def validate_dry_run(
    data: pd.DataFrame,
    starts: list[tuple[str, str, pd.Timestamp]],
    args: argparse.Namespace,
    weather_columns: list[str],
) -> None:
    print("Dry run: validating forecast windows without loading Chronos-2.")
    for index, (selection, week_id, forecast_start) in enumerate(starts, start=1):
        context_df, future_df, actual, info = build_window(data, forecast_start, args, weather_columns)
        print(
            f"[{index}/{len(starts)}] {selection} {week_id} {forecast_start}: "
            f"context_rows={len(context_df)}, future_rows={len(future_df)}, "
            f"actual_rows={len(actual)}, capped={info['context_capped']}"
        )
    print("Dry run completed.")


def run_one_forecast(
    pipeline,
    data: pd.DataFrame,
    selection: str,
    week_id: str,
    forecast_start: pd.Timestamp,
    args: argparse.Namespace,
    weather_columns: list[str],
    data_loading_seconds: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    forecast_loop_start = time.perf_counter()
    context_df, future_df, actual, context_info = build_window(data, forecast_start, args, weather_columns)
    print(
        f"Context rows: {len(context_df):,} "
        f"(requested {context_info['requested_context_steps']:,}, "
        f"effective {context_info['effective_context_steps']:,}, "
        f"capped={context_info['context_capped']})",
        flush=True,
    )
    if "target" in future_df.columns:
        raise ValueError("future_df must not contain target.")

    prediction_start = time.perf_counter()
    pred_df = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=prediction_steps(args.prediction_hours, args.resolution),
        quantile_levels=DEFAULT_QUANTILES,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
    )
    prediction_seconds = time.perf_counter() - prediction_start

    forecast = build_forecast_output(pred_df, actual, RESOLUTION_TO_STEP[args.resolution], DEFAULT_QUANTILES)
    forecast.insert(0, "model", MODEL_NAME)
    forecast.insert(1, "selection", selection)
    forecast.insert(2, "week_id", week_id)
    forecast.insert(3, "forecast_start", forecast_start.tz_localize(None))
    forecast.insert(4, "context_start", pd.Timestamp(context_info["context_start"]).tz_localize(None))
    forecast.insert(5, "requested_context_start", pd.Timestamp(context_info["requested_context_start"]).tz_localize(None))
    forecast.insert(6, "context_hours", args.context_hours)
    forecast.insert(7, "requested_context_hours", context_info["requested_context_hours"])
    forecast.insert(8, "effective_context_hours", context_info["effective_context_hours"])
    forecast.insert(9, "requested_context_steps", context_info["requested_context_steps"])
    forecast.insert(10, "effective_context_steps", context_info["effective_context_steps"])
    forecast.insert(11, "context_capped", context_info["context_capped"])
    forecast.insert(12, "resolution", args.resolution)
    forecast.insert(13, "prediction_hours", args.prediction_hours)
    forecast.insert(14, "prediction_steps", prediction_steps(args.prediction_hours, args.resolution))
    forecast.insert(15, "prediction_seconds", prediction_seconds)
    forecast.insert(16, "weather_columns", ",".join(weather_columns))

    metrics = calculate_forecast_metrics(
        forecast=forecast,
        args=args,
        selection=selection,
        week_id=week_id,
        forecast_start=forecast_start,
        data_loading_seconds=data_loading_seconds,
        prediction_seconds=prediction_seconds,
        context_info=context_info,
    )
    forecast_loop_seconds = time.perf_counter() - forecast_loop_start
    forecast["forecast_loop_seconds"] = forecast_loop_seconds
    metrics["forecast_loop_seconds"] = forecast_loop_seconds
    return forecast, metrics


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()
    weather_columns = parse_weather_columns(args.weather_columns)
    step = RESOLUTION_TO_STEP[args.resolution]

    selected_weeks = load_selected_weeks(args.selected_weeks_path)
    starts = forecast_starts_for_selected_weeks(
        selected_weeks,
        prediction_hours=args.prediction_hours,
        max_forecast_starts=args.max_forecast_starts,
    )
    if not starts:
        raise ValueError("No forecast starts were generated.")

    print(f"Model: {MODEL_NAME}")
    print(f"Model path: {args.model_path}")
    print(f"Resolution: {args.resolution}")
    print(f"Prediction hours: {args.prediction_hours}")
    print(f"Context hours requested: {args.context_hours}")
    print(f"Weather columns: {', '.join(weather_columns)}")
    print(f"Forecast starts requested: {len(starts):,}")
    print(f"Max context steps: {args.max_context_steps:,}")

    load_start = time.perf_counter()
    try:
        data = load_merged_data(args.heat_path, args.weather_path, weather_columns, step=step)
    except UnsupportedDataCadenceError as exc:
        raise SystemExit(str(exc)) from None
    validate_complete_year(
        data,
        year=2024,
        step=step,
        value_columns=["heat", *weather_columns],
        label="selected-week forecast data",
    )
    data = add_synthetic_model_clock(data, step)
    data_loading_seconds = time.perf_counter() - load_start
    print(f"Loaded merged data rows: {len(data):,}")

    if args.dry_run:
        validate_dry_run(data, starts, args, weather_columns)
        return

    feature_label = "_".join(weather_columns)
    cap_label = ""
    if requested_context_steps(args.context_hours, args.resolution) > args.max_context_steps:
        cap_label = f"_cap{args.max_context_steps}steps"
    default_run_name = (
        f"chronos2_extreme_{args.resolution}_pred{args.prediction_hours}h_"
        f"context{args.context_hours}h{cap_label}_{feature_label}"
    )
    run_id = make_run_id(args.run_id, args.run_name or default_run_name)
    run_dir = args.output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    raw_output_path = run_dir / "raw_predictions.csv"
    metrics_output_path = run_dir / "metrics_per_forecast_start.csv"
    summary_output_path = run_dir / "metrics_summary.csv"
    metadata_path = run_dir / "run_metadata.json"
    command_path = run_dir / "command.txt"
    command_path.write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    pipeline = initialize_pipeline(args)

    raw_forecasts = []
    metric_rows = []
    for forecast_number, (selection, week_id, forecast_start) in enumerate(starts, start=1):
        print(f"[{forecast_number}/{len(starts)}] Predicting {selection} {week_id}: {forecast_start}", flush=True)
        forecast, metrics = run_one_forecast(
            pipeline=pipeline,
            data=data,
            selection=selection,
            week_id=week_id,
            forecast_start=forecast_start,
            args=args,
            weather_columns=weather_columns,
            data_loading_seconds=data_loading_seconds,
        )
        raw_forecasts.append(forecast)
        metric_rows.append(metrics)
        print(
            f"[{forecast_number}/{len(starts)}] MAE={metrics['MAE']:.3f}, "
            f"RMSE={metrics['RMSE']:.3f}, "
            f"CVRMSE={metrics['CVRMSE_percent']:.3f}%, "
            f"prediction={metrics['prediction_seconds']:.2f}s",
            flush=True,
        )

    raw = pd.concat(raw_forecasts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = calculate_summary_metrics(raw)
    raw.to_csv(raw_output_path, index=False)
    metrics.to_csv(metrics_output_path, index=False)
    summary.to_csv(summary_output_path, index=False)

    total_seconds = time.perf_counter() - total_start
    write_metadata(
        metadata_path,
        build_metadata(
            args,
            run_id,
            run_dir,
            selected_weeks,
            weather_columns,
            total_seconds,
        ),
    )

    print(f"Completed forecast starts: {len(starts):,}")
    print(f"Saved raw predictions: {raw_output_path}")
    print(f"Saved per-start metrics: {metrics_output_path}")
    print(f"Saved summary metrics: {summary_output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Total seconds: {total_seconds:.2f}")


if __name__ == "__main__":
    main()
