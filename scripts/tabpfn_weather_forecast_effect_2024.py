from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from full_year_forecasting_utils import (
    HOURLY_STEP,
    ITEM_ID,
    add_error_columns,
    add_synthetic_model_clock,
    data_slice_by_year,
    flatten_tabpfn_prediction,
    forecast_starts_from_rows,
    load_heat_weather_comparison_data,
    metric_values,
    sha256_file,
    to_naive_datetime,
)
from tabpfn_ts_full_year_2024 import DEFAULT_QUANTILES
from utils import TABPFN_PACKAGES, make_run_id, metadata_envelope, write_metadata


DEFAULT_HEAT_PATH = Path("munich/demand/heat/heat_dh.csv")
DEFAULT_WEATHER_COMPARISON_PATH = Path("munich/weather/munich_temperature_observed_vs_forecast_2024.csv")
DEFAULT_WEATHER_METADATA_PATH = Path("munich/weather/munich_temperature_observed_vs_forecast_2024_metadata.json")
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/weather_forecast_effect_2024")
DEFAULT_RUN_NAME = "tabpfn_weather_forecast_effect_2024_hourly_pred24h_context12w"
WEATHER_MODES = ("observed_temperature", "forecast_temperature_24h")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TabPFN-TS full-year forecasts using observed future temperature "
            "versus archived 24h-ahead forecast temperature in future_df."
        )
    )
    parser.add_argument("--heat-path", type=Path, default=DEFAULT_HEAT_PATH)
    parser.add_argument("--weather-comparison-path", type=Path, default=DEFAULT_WEATHER_COMPARISON_PATH)
    parser.add_argument("--weather-metadata-path", type=Path, default=DEFAULT_WEATHER_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--context-hours", type=int, default=12 * 7 * 24)
    parser.add_argument("--prediction-hours", type=int, default=24)
    parser.add_argument("--forecast-every-hours", type=int, default=24)
    parser.add_argument("--mode", choices=("CLIENT", "LOCAL"), default="LOCAL")
    parser.add_argument("--max-forecast-starts", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and windows without running TabPFN-TS or writing outputs.",
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


def make_context_frame(context_raw: pd.DataFrame) -> pd.DataFrame:
    timestamp_source = "model_timestamp" if "model_timestamp" in context_raw.columns else "timestamp"
    out = pd.DataFrame(
        {
            "item_id": ITEM_ID,
            "timestamp": to_naive_datetime(context_raw[timestamp_source]),
            "target": context_raw["heat"],
            "temperature": context_raw["temperature_observed"],
        }
    )
    return out


def make_future_frame(future_raw: pd.DataFrame, weather_mode: str) -> pd.DataFrame:
    if weather_mode == "observed_temperature":
        temperature = future_raw["temperature_observed"]
    elif weather_mode == "forecast_temperature_24h":
        temperature = future_raw["temperature_forecast_24h"]
    else:
        raise ValueError(f"Unknown weather mode: {weather_mode}")
    timestamp_source = "model_timestamp" if "model_timestamp" in future_raw.columns else "timestamp"
    future = pd.DataFrame(
        {
            "item_id": ITEM_ID,
            "timestamp": to_naive_datetime(future_raw[timestamp_source]),
            "temperature": temperature,
        }
    )
    if "target" in future.columns:
        raise ValueError("future_df must not contain target.")
    return future


def build_window(
    data: pd.DataFrame,
    forecast_start: pd.Timestamp,
    context_steps: int,
    horizon_steps: int,
    weather_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = data.index[data["timestamp"].eq(forecast_start)].tolist()
    if len(matches) != 1:
        raise ValueError(f"Forecast start {forecast_start} not found exactly once.")
    start_pos = matches[0]
    if start_pos < context_steps:
        raise ValueError(
            f"Forecast start {forecast_start} has only {start_pos} rows of history, "
            f"expected {context_steps}."
        )
    context_raw = data.iloc[start_pos - context_steps : start_pos].copy()
    future_raw = data.iloc[start_pos : start_pos + horizon_steps].copy()
    if len(context_raw) != context_steps:
        raise ValueError(f"Context for {forecast_start} has {len(context_raw)} rows, expected {context_steps}.")
    if len(future_raw) != horizon_steps:
        raise ValueError(f"Future for {forecast_start} has {len(future_raw)} rows, expected {horizon_steps}.")
    context_required = ["heat", "temperature_observed"]
    if context_raw[context_required].isna().any().any():
        missing_counts = context_raw[context_required].isna().sum().to_dict()
        raise ValueError(f"Context for {forecast_start} contains missing values: {missing_counts}")
    future_required = ["heat", "temperature_observed"]
    if weather_mode == "forecast_temperature_24h":
        future_required.append("temperature_forecast_24h")
    if future_raw[future_required].isna().any().any():
        missing_counts = future_raw[future_required].isna().sum().to_dict()
        raise ValueError(f"Future for {forecast_start} and {weather_mode} contains missing values: {missing_counts}")
    context_df = make_context_frame(context_raw)
    future_df = make_future_frame(future_raw, weather_mode)
    actual_columns = ["timestamp"]
    if "model_timestamp" in future_raw.columns:
        actual_columns.append("model_timestamp")
    actual_columns.extend(["heat", "temperature_observed", "temperature_forecast_24h"])
    actual = future_raw[actual_columns].rename(columns={"heat": "actual_heat"})
    if "model_timestamp" in actual.columns:
        actual["model_timestamp"] = to_naive_datetime(actual["model_timestamp"])
    else:
        actual["timestamp"] = to_naive_datetime(actual["timestamp"])
    return context_df, future_df, actual


def is_usable_forecast_start(
    data: pd.DataFrame,
    forecast_start: pd.Timestamp,
    context_steps: int,
    horizon_steps: int,
) -> bool:
    try:
        for weather_mode in WEATHER_MODES:
            build_window(data, forecast_start, context_steps, horizon_steps, weather_mode)
    except ValueError:
        return False
    return True


def filter_usable_forecast_starts(
    data: pd.DataFrame,
    starts: pd.DataFrame,
    context_steps: int,
    horizon_steps: int,
) -> pd.DataFrame:
    usable_mask = [
        is_usable_forecast_start(data, row["forecast_start"], context_steps, horizon_steps)
        for _, row in starts.iterrows()
    ]
    return starts.loc[usable_mask].reset_index(drop=True)


def predict_one(
    pipeline,
    data: pd.DataFrame,
    forecast_start: pd.Timestamp,
    context_steps: int,
    horizon_steps: int,
    weather_mode: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    context_df, future_df, actual = build_window(data, forecast_start, context_steps, horizon_steps, weather_mode)
    if getattr(pipeline, "max_context_length", 0) < len(context_df):
        raise ValueError(
            "TabPFN-TS would truncate context: "
            f"max_context_length={pipeline.max_context_length}, context rows={len(context_df)}."
        )
    started = time.perf_counter()
    prediction = pipeline.predict_df(context_df=context_df, future_df=future_df, quantiles=DEFAULT_QUANTILES)
    prediction_seconds = time.perf_counter() - started

    predicted = flatten_tabpfn_prediction(prediction, DEFAULT_QUANTILES)
    if "model_timestamp" in actual.columns:
        predicted = predicted.rename(columns={"timestamp": "model_timestamp"})
        predicted["model_timestamp"] = pd.to_datetime(predicted["model_timestamp"], errors="raise")
        forecast = actual.merge(predicted, on="model_timestamp", how="left", validate="one_to_one")
        forecast = forecast.drop(columns=["model_timestamp"])
    else:
        forecast = actual.merge(predicted, on="timestamp", how="left", validate="one_to_one")
    if forecast["predicted_heat"].isna().any():
        raise ValueError(f"Missing predictions for {weather_mode} at {forecast_start}.")
    forecast.insert(0, "model", "TabPFN-TS")
    forecast.insert(1, "weather_mode", weather_mode)
    forecast.insert(2, "forecast_start", forecast_start.isoformat())
    forecast.insert(3, "horizon_step", np.arange(1, len(forecast) + 1))
    forecast.insert(4, "horizon_minutes", forecast["horizon_step"] * 60)
    forecast["resolution"] = "hourly"
    forecast["context_hours"] = context_steps
    forecast["prediction_hours"] = horizon_steps
    forecast["prediction_steps"] = horizon_steps
    forecast["prediction_seconds"] = prediction_seconds
    forecast["context_rows"] = len(context_df)
    forecast = add_error_columns(forecast)

    metrics: dict[str, object] = {
        "model": "TabPFN-TS",
        "weather_mode": weather_mode,
        "forecast_start": forecast_start.isoformat(),
        "resolution": "hourly",
        "context_hours": context_steps,
        "prediction_hours": horizon_steps,
        "prediction_steps": horizon_steps,
        "n_context_rows": len(context_df),
        "n_forecast_rows": len(forecast),
        "prediction_seconds": prediction_seconds,
    }
    metrics.update(metric_values(forecast["actual_heat"], forecast["predicted_heat"]))
    return forecast, metrics


def calculate_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for weather_mode, group in raw.groupby("weather_mode", sort=True):
        row: dict[str, object] = {
            "model": "TabPFN-TS",
            "weather_mode": weather_mode,
            "metric_scope": "all",
            "n_forecast_starts": int(group["forecast_start"].nunique()),
            "n_rows": int(len(group)),
        }
        row.update(metric_values(group["actual_heat"], group["predicted_heat"]))
        per_start = group.groupby("forecast_start", sort=False)["prediction_seconds"].first()
        row["prediction_seconds_total"] = float(per_start.sum())
        row["prediction_seconds_mean_per_forecast_start"] = float(per_start.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_metadata(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    n_candidate_forecast_starts: int,
    n_usable_forecast_starts: int,
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
            "heat_path": str(args.heat_path),
            "heat_sha256": sha256_file(args.heat_path),
            "weather_comparison_path": str(args.weather_comparison_path),
            "weather_comparison_sha256": sha256_file(args.weather_comparison_path),
            "weather_metadata_path": str(args.weather_metadata_path),
            "weather_metadata_sha256": sha256_file(args.weather_metadata_path),
        },
        "experiment": {
            "purpose": "Sensitivity test for perfect future weather assumption.",
            "year": args.year,
            "resolution": "hourly",
            "context_hours": args.context_hours,
            "prediction_hours": args.prediction_hours,
            "forecast_every_hours": args.forecast_every_hours,
            "weather_modes": {
                "observed_temperature": "future_df.temperature uses observed future temperature",
                "forecast_temperature_24h": "future_df.temperature uses archived 24h-ahead forecast temperature",
            },
            "context_temperature": "observed temperature in both modes",
            "future_covariates_exclude_target": True,
            "timestamp_axis": (
                "TabPFN-TS receives a synthetic continuous hourly model clock. "
                "Real Europe/Berlin timestamps are retained in outputs. This avoids "
                "duplicate or missing naive timestamps at daylight-saving-time transitions."
            ),
            "forecast_start_filter": (
                "Keeps only starts with complete context, complete target horizons, "
                "complete observed future temperature, and complete 24h-ahead forecast temperature."
            ),
            "n_candidate_forecast_starts": n_candidate_forecast_starts,
            "n_usable_forecast_starts_per_mode": n_usable_forecast_starts,
        },
        "quantiles": DEFAULT_QUANTILES,
        "total_seconds": total_seconds,
    })
    return doc


def main() -> None:
    args = parse_args()
    context_steps = args.context_hours
    horizon_steps = args.prediction_hours
    stride_steps = args.forecast_every_hours
    data = load_heat_weather_comparison_data(args.heat_path, args.weather_comparison_path)
    data = add_synthetic_model_clock(data, HOURLY_STEP)
    candidate_starts = forecast_starts_from_rows(
        data,
        args.year,
        horizon_steps,
        stride_steps,
        value_columns=["heat", "temperature_observed"],
    )
    if candidate_starts.empty:
        raise ValueError(f"No forecast starts found for {args.year}.")
    starts = filter_usable_forecast_starts(data, candidate_starts, context_steps, horizon_steps)
    if args.max_forecast_starts is not None:
        starts = starts.head(args.max_forecast_starts)
    if starts.empty:
        year_data = data_slice_by_year(data, args.year)
        n_missing_forecast = int(year_data["temperature_forecast_24h"].isna().sum())
        raise ValueError(
            "No usable forecast starts remain after filtering for complete 24h forecast-temperature horizons. "
            f"Missing forecast-temperature rows in {args.year}: {n_missing_forecast}."
        )

    print(f"Candidate forecast starts: {len(candidate_starts):,}")
    print(f"Usable forecast starts per weather mode: {len(starts):,}")
    print(f"First start: {starts['forecast_start'].iloc[0]}")
    print(f"Last start: {starts['forecast_start'].iloc[-1]}")
    if args.dry_run:
        for weather_mode in WEATHER_MODES:
            for forecast_start in starts["forecast_start"]:
                build_window(data, forecast_start, context_steps, horizon_steps, weather_mode)
        print("Dry run complete. No predictions or outputs written.")
        return

    total_start = time.perf_counter()
    run_id = make_run_id(args.run_id, args.run_name)
    run_dir = args.output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    pipeline = initialize_pipeline(args.mode, max_context_length=context_steps)

    raw_forecasts = []
    metric_rows = []
    total_predictions = len(starts) * len(WEATHER_MODES)
    counter = 0
    for weather_mode in WEATHER_MODES:
        for _, row in starts.iterrows():
            counter += 1
            forecast_start = row["forecast_start"]
            print(f"[{counter}/{total_predictions}] {weather_mode}: {forecast_start}")
            forecast, metrics = predict_one(
                pipeline=pipeline,
                data=data,
                forecast_start=forecast_start,
                context_steps=context_steps,
                horizon_steps=horizon_steps,
                weather_mode=weather_mode,
            )
            raw_forecasts.append(forecast)
            metric_rows.append(metrics)
            print(
                f"  RMSE={metrics['RMSE']:.3f}, CVRMSE={metrics['CVRMSE_percent']:.2f}%, "
                f"prediction={metrics['prediction_seconds']:.2f}s"
            )

    raw = pd.concat(raw_forecasts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = calculate_summary(raw)
    raw.to_csv(run_dir / "raw_predictions.csv", index=False)
    metrics.to_csv(run_dir / "metrics_per_forecast_start.csv", index=False)
    summary.to_csv(run_dir / "metrics_summary.csv", index=False)

    total_seconds = time.perf_counter() - total_start
    write_metadata(
        run_dir / "run_metadata.json",
        build_metadata(
            args,
            run_id,
            run_dir,
            len(candidate_starts),
            len(starts),
            total_seconds,
        ),
    )
    print(f"Saved run directory: {run_dir}")
    print(f"Saved raw predictions: {run_dir / 'raw_predictions.csv'}")
    print(f"Saved per-start metrics: {run_dir / 'metrics_per_forecast_start.csv'}")
    print(f"Saved summary metrics: {run_dir / 'metrics_summary.csv'}")
    print(f"Saved metadata: {run_dir / 'run_metadata.json'}")
    print(f"Total seconds: {total_seconds:.2f}")


if __name__ == "__main__":
    main()
