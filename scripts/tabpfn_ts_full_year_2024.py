from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from full_year_forecasting_utils import (
    HOURLY_STEP,
    QUARTER_STEP,
    STEPS_PER_HOUR,
    UnsupportedDataCadenceError,
    add_error_columns,
    add_synthetic_model_clock,
    build_tabpfn_window,
    calculate_summary,
    flatten_tabpfn_prediction,
    forecast_starts_from_rows,
    load_merged_data,
    metric_values,
    sha256_file,
    validate_regular_rows,
)
from utils import (
    TABPFN_PACKAGES,
    make_run_id,
    metadata_envelope,
    parse_weather_columns,
    write_metadata,
)


DEFAULT_HEAT_PATH = Path("flensburg/demand/heat/heat_dh.csv")
DEFAULT_WEATHER_PATH = Path("flensburg/weather/flensburg_weather_temperature.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/full_year_2024/tabpfn")
DEFAULT_RUN_NAME = "tabpfn_full_year_2024_hourly_pred24h_context12w_temperature"
RESOLUTION_STEPS = {"hourly": 1, "quarter": STEPS_PER_HOUR}
RESOLUTION_TIMESTEPS = {"hourly": HOURLY_STEP, "quarter": QUARTER_STEP}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TabPFN-TS forecasts over a full calendar year. Default is the "
            "Flensburg 2024 setup: hourly data, 12-week context, 24h horizon, "
            "one forecast every 24 hourly rows, and ambient temperature only."
        )
    )
    parser.add_argument("--heat-path", type=Path, default=DEFAULT_HEAT_PATH)
    parser.add_argument("--weather-path", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--dataset-name", default="flensburg")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--resolution",
        choices=("hourly", "quarter"),
        default="hourly",
        help="hourly aggregates complete hours; quarter uses native 15-minute data.",
    )
    parser.add_argument("--weather-columns", default="temperature")
    parser.add_argument("--context-hours", type=int, default=12 * 7 * 24)
    parser.add_argument("--prediction-hours", type=int, default=24)
    parser.add_argument("--forecast-every-hours", type=int, default=24)
    parser.add_argument("--mode", choices=("CLIENT", "LOCAL"), default="LOCAL")
    parser.add_argument("--max-forecast-starts", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and forecast-start construction without running predictions or writing outputs.",
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


def predict_one_start(
    pipeline,
    data: pd.DataFrame,
    forecast_start: pd.Timestamp,
    context_steps: int,
    context_hours: int,
    horizon_steps: int,
    prediction_hours: int,
    resolution: str,
    step_minutes: int,
    weather_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    context_df, future_df, actual = build_tabpfn_window(
        data=data,
        forecast_start=forecast_start,
        context_steps=context_steps,
        horizon_steps=horizon_steps,
        target_column="heat",
        covariate_columns=weather_columns,
    )
    if getattr(pipeline, "max_context_length", 0) < len(context_df):
        raise ValueError(
            "TabPFN-TS would truncate context: "
            f"max_context_length={pipeline.max_context_length}, context rows={len(context_df)}."
        )
    if "target" in future_df.columns:
        raise ValueError("future_df must not contain target.")

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
        raise ValueError(f"Missing predictions for forecast start {forecast_start}.")
    forecast.insert(0, "model", "TabPFN-TS")
    forecast.insert(1, "forecast_start", forecast_start.isoformat())
    forecast.insert(2, "horizon_step", np.arange(1, len(forecast) + 1))
    forecast.insert(3, "horizon_minutes", forecast["horizon_step"] * step_minutes)
    forecast["resolution"] = resolution
    forecast["context_hours"] = context_hours
    forecast["prediction_hours"] = prediction_hours
    forecast["prediction_steps"] = horizon_steps
    forecast["prediction_seconds"] = prediction_seconds
    forecast["context_rows"] = len(context_df)
    forecast = add_error_columns(forecast)

    metrics: dict[str, object] = {
        "model": "TabPFN-TS",
        "forecast_start": forecast_start.isoformat(),
        "resolution": resolution,
        "context_hours": context_hours,
        "prediction_hours": prediction_hours,
        "prediction_steps": horizon_steps,
        "n_context_rows": len(context_df),
        "n_forecast_rows": len(forecast),
        "prediction_seconds": prediction_seconds,
    }
    metrics.update(metric_values(forecast["actual_heat"], forecast["predicted_heat"]))
    return forecast, metrics


def build_metadata(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    weather_columns: list[str],
    n_forecast_starts: int,
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
            "weather_columns": weather_columns,
        },
        "forecasting": {
            "year": args.year,
            "resolution": args.resolution,
            "context_hours": args.context_hours,
            "context_rows": args.context_hours * RESOLUTION_STEPS[args.resolution],
            "prediction_hours": args.prediction_hours,
            "prediction_steps": args.prediction_hours * RESOLUTION_STEPS[args.resolution],
            "forecast_every_hours": args.forecast_every_hours,
            "forecast_every_rows": args.forecast_every_hours * RESOLUTION_STEPS[args.resolution],
            "forecast_cadence": "row-based over the evaluation year",
            "timestamp_axis": (
                "TabPFN-TS receives a synthetic continuous model clock at the selected resolution. "
                "Real Europe/Berlin timestamps are retained in outputs. This avoids "
                "duplicate or missing naive timestamps at daylight-saving-time transitions."
            ),
            "mode": args.mode,
            "n_forecast_starts": n_forecast_starts,
            "future_covariates_exclude_target": True,
        },
        "quantiles": DEFAULT_QUANTILES,
        "total_seconds": total_seconds,
    })
    return doc


def main() -> None:
    args = parse_args()
    weather_columns = parse_weather_columns(args.weather_columns)
    resolution_steps = RESOLUTION_STEPS[args.resolution]
    step = RESOLUTION_TIMESTEPS[args.resolution]
    step_minutes = int(step / pd.Timedelta(minutes=1))
    horizon_steps = args.prediction_hours * resolution_steps
    context_steps = args.context_hours * resolution_steps
    stride_steps = args.forecast_every_hours * resolution_steps

    try:
        data = load_merged_data(args.heat_path, args.weather_path, weather_columns, step=step)
    except UnsupportedDataCadenceError as exc:
        raise SystemExit(str(exc)) from None
    validate_regular_rows(data, step, f"{args.resolution} merged data")
    data = add_synthetic_model_clock(data, step)
    starts = forecast_starts_from_rows(
        data=data,
        year=args.year,
        horizon_steps=horizon_steps,
        stride_steps=stride_steps,
    )
    if args.max_forecast_starts is not None:
        starts = starts.head(args.max_forecast_starts)
    if starts.empty:
        raise ValueError(f"No forecast starts found for {args.year}.")

    print(f"Dataset: {args.dataset_name}")
    print(f"Resolution: {args.resolution}")
    print(f"Weather columns: {', '.join(weather_columns)}")
    print(f"Context: {args.context_hours:,}h = {context_steps:,} rows")
    print(f"Prediction horizon: {args.prediction_hours:,}h = {horizon_steps:,} rows")
    print(f"Forecast cadence: {args.forecast_every_hours:,}h = {stride_steps:,} rows")
    print(f"Forecast starts: {len(starts):,}")
    print(f"First start: {starts['forecast_start'].iloc[0]}")
    print(f"Last start: {starts['forecast_start'].iloc[-1]}")

    if args.dry_run:
        for forecast_start in starts["forecast_start"]:
            build_tabpfn_window(
                data,
                forecast_start=forecast_start,
                context_steps=context_steps,
                horizon_steps=horizon_steps,
                target_column="heat",
                covariate_columns=weather_columns,
            )
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
    for idx, row in starts.iterrows():
        forecast_start = row["forecast_start"]
        print(f"[{idx + 1}/{len(starts)}] Forecast start {forecast_start}")
        forecast, metrics = predict_one_start(
            pipeline=pipeline,
            data=data,
            forecast_start=forecast_start,
            context_steps=context_steps,
            context_hours=args.context_hours,
            horizon_steps=horizon_steps,
            prediction_hours=args.prediction_hours,
            resolution=args.resolution,
            step_minutes=step_minutes,
            weather_columns=weather_columns,
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
            args=args,
            run_id=run_id,
            run_dir=run_dir,
            weather_columns=weather_columns,
            n_forecast_starts=len(starts),
            total_seconds=total_seconds,
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
