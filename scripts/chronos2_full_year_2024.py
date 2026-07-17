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
    UnsupportedDataCadenceError,
    add_error_columns,
    add_synthetic_model_clock,
    calculate_summary,
    forecast_starts_from_rows,
    load_merged_data,
    metric_values,
    safe_name,
    sha256_file,
    to_naive_datetime,
    validate_regular_rows,
)
from utils import (
    CHRONOS_PACKAGES,
    initialize_wandb,
    make_run_id,
    metadata_envelope,
    parse_weather_columns,
    write_metadata,
)


DEFAULT_HEAT_PATH = Path("flensburg/demand/heat/heat_dh.csv")
DEFAULT_WEATHER_PATH = Path("flensburg/weather/flensburg_weather_temperature.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/full_year_2024/chronos2")
DEFAULT_RUN_NAME = "chronos2_full_year_2024_hourly_pred24h_context12w_temperature"
DEFAULT_WANDB_PROJECT = "timeseries-forecasting"
DEFAULT_MODEL_PATH = "amazon/chronos-2"
DEFAULT_MAX_CONTEXT_STEPS = 8192
MODEL_NAME = "Chronos-2"
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
            "Run direct Chronos-2 forecasts over a full calendar year. Default is the "
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
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--max-context-steps", type=int, default=DEFAULT_MAX_CONTEXT_STEPS)
    parser.add_argument("--max-forecast-starts", type=int, default=None)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and forecast-start construction without running predictions or writing outputs.",
    )
    return parser.parse_args()


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


def requested_context_steps(context_hours: int, resolution: str) -> int:
    if context_hours <= 0:
        raise ValueError(f"context-hours must be positive, got {context_hours}.")
    return context_hours * RESOLUTION_STEPS[resolution]


def effective_context_steps(context_hours: int, resolution: str, max_context_steps: int) -> int:
    if max_context_steps <= 0:
        raise ValueError(f"max-context-steps must be positive, got {max_context_steps}.")
    return min(requested_context_steps(context_hours, resolution), max_context_steps)


def initialize_pipeline(args: argparse.Namespace):
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install Chronos-2 direct inference support with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return Chronos2Pipeline.from_pretrained(args.model_path, device_map=args.device_map)


def log_wandb_summary(wandb_run, summary: pd.DataFrame) -> None:
    if wandb_run is None:
        return
    payload = {}
    for _, row in summary.iterrows():
        scope = safe_name(str(row["metric_scope"]))
        for key in ["MAE", "RMSE", "R2", "CVRMSE_percent", "sMAPE_percent"]:
            if key in row and pd.notna(row[key]):
                payload[f"summary/{scope}/{key}"] = float(row[key])
    if payload:
        wandb_run.log(payload)


def make_chronos_frame(
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


def build_chronos_window(
    data: pd.DataFrame,
    forecast_start: pd.Timestamp,
    context_steps: int,
    requested_steps: int,
    horizon_steps: int,
    target_column: str,
    covariate_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    matches = data.index[data["timestamp"].eq(forecast_start)].tolist()
    if len(matches) != 1:
        raise ValueError(f"Forecast start {forecast_start} not found exactly once.")
    start_pos = matches[0]
    context_start_pos = start_pos - context_steps
    horizon_end_pos = start_pos + horizon_steps
    if context_start_pos < 0:
        raise ValueError(f"Not enough context before {forecast_start}.")
    if horizon_end_pos > len(data):
        raise ValueError(f"Not enough future rows after {forecast_start}.")

    context_raw = data.iloc[context_start_pos:start_pos].copy()
    future_raw = data.iloc[start_pos:horizon_end_pos].copy()
    required_context = [target_column, *covariate_columns]
    if context_raw[required_context].isna().any().any():
        missing_counts = context_raw[required_context].isna().sum().to_dict()
        raise ValueError(f"Missing context values at {forecast_start}: {missing_counts}")
    if future_raw[covariate_columns].isna().any().any():
        missing_counts = future_raw[covariate_columns].isna().sum().to_dict()
        raise ValueError(f"Missing future covariates at {forecast_start}: {missing_counts}")

    context_df = make_chronos_frame(context_raw, target_column, covariate_columns, include_target=True)
    future_df = make_chronos_frame(future_raw, target_column, covariate_columns, include_target=False)
    if "target" in future_df.columns:
        raise ValueError("future_df must not contain target.")

    actual_columns = ["timestamp"]
    if "model_timestamp" in future_raw.columns:
        actual_columns.append("model_timestamp")
    actual_columns.extend([target_column, *covariate_columns])
    actual = future_raw[actual_columns].rename(columns={target_column: "actual_heat"})
    if "model_timestamp" in actual.columns:
        actual["model_timestamp"] = to_naive_datetime(actual["model_timestamp"])
    else:
        actual["timestamp"] = to_naive_datetime(actual["timestamp"])
    info = {
        "requested_context_steps": requested_steps,
        "effective_context_steps": context_steps,
        "context_capped": requested_steps > context_steps,
    }
    return context_df, future_df, actual, info


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


def predict_one_start(
    pipeline,
    data: pd.DataFrame,
    forecast_start: pd.Timestamp,
    context_steps: int,
    requested_steps: int,
    context_hours: int,
    horizon_steps: int,
    prediction_hours: int,
    resolution: str,
    step_minutes: int,
    weather_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    context_df, future_df, actual, context_info = build_chronos_window(
        data=data,
        forecast_start=forecast_start,
        context_steps=context_steps,
        requested_steps=requested_steps,
        horizon_steps=horizon_steps,
        target_column="heat",
        covariate_columns=weather_columns,
    )
    if "target" in future_df.columns:
        raise ValueError("future_df must not contain target.")

    started = time.perf_counter()
    prediction = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=horizon_steps,
        quantile_levels=DEFAULT_QUANTILES,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
    )
    prediction_seconds = time.perf_counter() - started

    predicted = flatten_chronos_predictions(prediction, DEFAULT_QUANTILES)
    forecast = actual.merge(predicted, on="model_timestamp", how="left", validate="one_to_one")
    forecast = forecast.drop(columns=["model_timestamp"])
    if forecast["predicted_heat"].isna().any():
        raise ValueError(f"Missing predictions for forecast start {forecast_start}.")
    forecast.insert(0, "model", MODEL_NAME)
    forecast.insert(1, "forecast_start", forecast_start.isoformat())
    forecast.insert(2, "horizon_step", np.arange(1, len(forecast) + 1))
    forecast.insert(3, "horizon_minutes", forecast["horizon_step"] * step_minutes)
    forecast["resolution"] = resolution
    forecast["context_hours"] = context_hours
    forecast["requested_context_steps"] = context_info["requested_context_steps"]
    forecast["effective_context_steps"] = context_info["effective_context_steps"]
    forecast["context_capped"] = context_info["context_capped"]
    forecast["prediction_hours"] = prediction_hours
    forecast["prediction_steps"] = horizon_steps
    forecast["prediction_seconds"] = prediction_seconds
    forecast["context_rows"] = len(context_df)
    forecast = add_error_columns(forecast)

    metrics: dict[str, object] = {
        "model": MODEL_NAME,
        "forecast_start": forecast_start.isoformat(),
        "resolution": resolution,
        "context_hours": context_hours,
        "requested_context_steps": context_info["requested_context_steps"],
        "effective_context_steps": context_info["effective_context_steps"],
        "context_capped": context_info["context_capped"],
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
    req_steps = requested_context_steps(args.context_hours, args.resolution)
    eff_steps = effective_context_steps(args.context_hours, args.resolution, args.max_context_steps)
    doc = metadata_envelope(
        run_id=run_id,
        run_name=args.run_name,
        run_dir=run_dir,
        script_path=__file__,
        packages=CHRONOS_PACKAGES,
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
        "model": {
            "name": MODEL_NAME,
            "implementation": "chronos.Chronos2Pipeline",
            "model_path": args.model_path,
            "direct_chronos": True,
            "autogluon_used": False,
        },
        "forecasting": {
            "year": args.year,
            "resolution": args.resolution,
            "context_hours": args.context_hours,
            "requested_context_rows": req_steps,
            "effective_context_rows": eff_steps,
            "max_context_steps": args.max_context_steps,
            "context_capped": req_steps > eff_steps,
            "prediction_hours": args.prediction_hours,
            "prediction_steps": args.prediction_hours * RESOLUTION_STEPS[args.resolution],
            "forecast_every_hours": args.forecast_every_hours,
            "forecast_every_rows": args.forecast_every_hours * RESOLUTION_STEPS[args.resolution],
            "forecast_cadence": "row-based over the evaluation year",
            "timestamp_axis": (
                "Chronos-2 receives a synthetic continuous model clock at the selected resolution. "
                "Real Europe/Berlin timestamps are retained in outputs. This avoids "
                "duplicate or missing naive timestamps at daylight-saving-time transitions."
            ),
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
    req_context_steps = requested_context_steps(args.context_hours, args.resolution)
    eff_context_steps = effective_context_steps(args.context_hours, args.resolution, args.max_context_steps)
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
    print(f"Model: {MODEL_NAME}")
    print(f"Model path: {args.model_path}")
    print(f"Resolution: {args.resolution}")
    print(f"Weather columns: {', '.join(weather_columns)}")
    print(f"Requested context: {args.context_hours:,}h = {req_context_steps:,} rows")
    print(f"Effective context rows: {eff_context_steps:,} (capped={req_context_steps > eff_context_steps})")
    print(f"Prediction horizon: {args.prediction_hours:,}h = {horizon_steps:,} rows")
    print(f"Forecast cadence: {args.forecast_every_hours:,}h = {stride_steps:,} rows")
    print(f"Forecast starts: {len(starts):,}")
    print(f"First start: {starts['forecast_start'].iloc[0]}")
    print(f"Last start: {starts['forecast_start'].iloc[-1]}")

    if args.dry_run:
        for forecast_start in starts["forecast_start"]:
            build_chronos_window(
                data,
                forecast_start=forecast_start,
                context_steps=eff_context_steps,
                requested_steps=req_context_steps,
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

    pipeline = initialize_pipeline(args)
    wandb_run = initialize_wandb(
        args,
        run_id,
        {
            "run_id": run_id,
            "run_name": args.run_name,
            "model": MODEL_NAME,
            "model_path": args.model_path,
            "dataset_name": args.dataset_name,
            "year": args.year,
            "resolution": args.resolution,
            "context_hours": args.context_hours,
            "requested_context_steps": requested_context_steps(args.context_hours, args.resolution),
            "max_context_steps": args.max_context_steps,
            "effective_context_steps": effective_context_steps(
                args.context_hours,
                args.resolution,
                args.max_context_steps,
            ),
            "prediction_hours": args.prediction_hours,
            "forecast_every_hours": args.forecast_every_hours,
            "weather_columns": weather_columns,
        },
    )

    raw_forecasts = []
    metric_rows = []
    for idx, row in starts.iterrows():
        forecast_start = row["forecast_start"]
        print(f"[{idx + 1}/{len(starts)}] Forecast start {forecast_start}", flush=True)
        forecast, metrics = predict_one_start(
            pipeline=pipeline,
            data=data,
            forecast_start=forecast_start,
            context_steps=eff_context_steps,
            requested_steps=req_context_steps,
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
            f"prediction={metrics['prediction_seconds']:.2f}s",
            flush=True,
        )

    raw = pd.concat(raw_forecasts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = calculate_summary(raw)

    raw.to_csv(run_dir / "raw_predictions.csv", index=False)
    metrics.to_csv(run_dir / "metrics_per_forecast_start.csv", index=False)
    summary.to_csv(run_dir / "metrics_summary.csv", index=False)

    if wandb_run is not None:
        log_wandb_summary(wandb_run, summary)
        wandb_run.finish()

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
