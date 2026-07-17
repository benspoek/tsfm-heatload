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
    forecast_starts_for_selected_weeks,
    load_selected_weeks,
    safe_name,
    sha256_file,
    validate_complete_year,
)
from tabpfn_ts_heat_forecast import (
    DEFAULT_HEAT_PATH,
    DEFAULT_OUTPUT_DIR as BASE_OUTPUT_DIR,
    DEFAULT_PREDICTION_HOURS,
    DEFAULT_WEATHER_PATH,
    EXPECTED_STEP_BY_RESOLUTION,
    RESOLUTION_TO_FREQ,
    STEPS_PER_HOUR_BY_RESOLUTION,
    TIMEZONE,
    WEATHER_COLUMNS,
    build_windows,
    find_column_by_numeric_value,
    load_data,
    prediction_steps,
    validate_model_inputs,
)
from utils import TABPFN_PACKAGES, initialize_wandb, make_run_id, metadata_envelope, write_metadata


DEFAULT_CONTEXT_HOURS = 365 * 24
DEFAULT_EXPERIMENT_OUTPUT_DIR = BASE_OUTPUT_DIR / "experiments/extreme_weeks_2024"
DEFAULT_WANDB_PROJECT = "timeseries-forecasting"
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
            "Run daily TabPFN-TS heat-demand backtests for the hottest, coldest, "
            "and most temperature-variable complete ISO weeks in 2024."
        )
    )
    parser.add_argument("--heat-path", type=Path, default=DEFAULT_HEAT_PATH)
    parser.add_argument("--weather-path", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--selected-weeks-path", type=Path, default=DEFAULT_SELECTED_WEEKS_PATH)
    parser.add_argument("--context-hours", type=int, default=DEFAULT_CONTEXT_HOURS)
    parser.add_argument(
        "--resolution",
        choices=tuple(STEPS_PER_HOUR_BY_RESOLUTION),
        required=True,
        help="Data resolution. quarter uses 15-minute data; hourly aggregates to hourly means.",
    )
    parser.add_argument(
        "--prediction-hours",
        type=int,
        default=DEFAULT_PREDICTION_HOURS,
        help="Forecast horizon in hours. Row count is calculated from --resolution.",
    )
    parser.add_argument(
        "--mode",
        choices=("CLIENT", "LOCAL"),
        default="CLIENT",
        help="TabPFN-TS inference mode.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None, help="Optional fixed run directory name.")
    parser.add_argument("--run-name", default=None, help="Optional human-readable run label.")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--max-forecast-starts", type=int, default=None)
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Run without logging metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and every forecast window without loading TabPFN-TS or writing outputs.",
    )
    return parser.parse_args()


def initialize_pipeline(mode: str, max_context_length: int):
    try:
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    return TabPFNTSPipeline(
        tabpfn_mode=getattr(TabPFNMode, mode),
        max_context_length=max_context_length,
    )


def quantile_column_name(value: float) -> str:
    percent = value * 100
    text = f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "_")
    if percent < 10 and not text.startswith("0"):
        text = f"0{text}"
    return f"q{text}"


def flatten_prediction_quantiles(pred_df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    out = pred_df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    elif "timestamp" not in out.columns:
        out = out.reset_index()

    if "timestamp" not in out.columns:
        raise ValueError(f"Prediction output has no timestamp column: {out.columns.tolist()}")
    if "target" not in out.columns:
        raise ValueError(f"Prediction output has no target column: {out.columns.tolist()}")

    result = out[["timestamp", "target"]].rename(columns={"target": "predicted_heat"}).copy()
    for quantile in quantiles:
        target_column = quantile_column_name(quantile)
        source_column = find_column_by_numeric_value(out.columns, quantile)
        if source_column is not None:
            result[target_column] = out[source_column]
        elif math.isclose(quantile, 0.5, rel_tol=0.0, abs_tol=1e-12):
            result[target_column] = result["predicted_heat"]
        else:
            result[target_column] = np.nan

    if "q50" in result.columns:
        result["predicted_heat"] = result["q50"]
    return result


def build_forecast_output(pred_df: pd.DataFrame, test_df: pd.DataFrame, step: pd.Timedelta, quantiles: list[float]) -> pd.DataFrame:
    pred = flatten_prediction_quantiles(pred_df, quantiles)
    out = test_df.merge(pred, on="timestamp", how="left", validate="one_to_one")

    if out["predicted_heat"].isna().any():
        raise ValueError("Some forecast timestamps did not receive predictions.")

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
        inside = (actual >= lower) & (actual <= upper)
        width = upper - lower
        metrics[f"coverage_{label}_percent"] = float(inside.mean() * 100)
        metrics[f"mean_width_{label}"] = float(width.mean())


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


def calculate_forecast_metrics(
    forecast: pd.DataFrame,
    args: argparse.Namespace,
    selection: str,
    week_id: str,
    forecast_start: pd.Timestamp,
    data_loading_seconds: float,
    prediction_seconds: float,
    n_context_rows: int,
) -> dict[str, object]:
    error = forecast["error"].to_numpy(dtype=float)
    actual = forecast["actual_heat"].to_numpy(dtype=float)
    absolute_error = np.abs(error)
    rmse = math.sqrt(float(np.mean(error**2)))
    smape = float(np.nanmean(forecast["sape"]) * 100)
    total_sum_of_squares = float(np.sum((actual - np.mean(actual)) ** 2))
    residual_sum_of_squares = float(np.sum(error**2))
    r2_score = (
        float(1 - residual_sum_of_squares / total_sum_of_squares)
        if total_sum_of_squares > 0
        else np.nan
    )
    mean_actual = float(np.mean(actual))

    metrics: dict[str, object] = {
        "selection": selection,
        "week_id": week_id,
        "forecast_start": forecast_start.tz_localize(None).isoformat(),
        "resolution": args.resolution,
        "context_hours": args.context_hours,
        "context_days_equivalent": args.context_hours / 24,
        "prediction_hours": args.prediction_hours,
        "prediction_steps": prediction_steps(args.prediction_hours, args.resolution),
        "mode": args.mode,
        "n_context_rows": n_context_rows,
        "n_forecast_rows": len(forecast),
        "MAE": float(np.mean(absolute_error)),
        "RMSE": rmse,
        "R2": r2_score,
        "CVRMSE_percent": float(rmse / mean_actual * 100) if mean_actual != 0 else np.nan,
        "sMAPE_percent": smape,
        "mean_error": float(np.mean(error)),
        "median_absolute_error": float(np.median(absolute_error)),
        "max_absolute_error": float(np.max(absolute_error)),
        "data_loading_seconds": data_loading_seconds,
        "prediction_seconds": prediction_seconds,
        "total_seconds": data_loading_seconds + prediction_seconds,
    }
    add_interval_metrics(metrics, forecast)

    for quantile in DEFAULT_QUANTILES:
        column = quantile_column_name(quantile)
        if column in forecast.columns:
            metrics[f"pinball_loss_{column}"] = pinball_loss(actual, forecast[column].to_numpy(dtype=float), quantile)
    return metrics


def calculate_group_metrics(scope: str, group: pd.DataFrame) -> dict[str, object]:
    error = group["error"].to_numpy(dtype=float)
    actual = group["actual_heat"].to_numpy(dtype=float)
    absolute_error = np.abs(error)
    rmse = math.sqrt(float(np.mean(error**2)))
    actual_mean = float(np.mean(actual))
    total_sum_of_squares = float(np.sum((actual - actual_mean) ** 2))
    residual_sum_of_squares = float(np.sum(error**2))
    metrics: dict[str, object] = {
        "metric_scope": scope,
        "n_forecast_starts": int(group["forecast_start"].nunique()),
        "n_rows": int(len(group)),
        "resolution": ",".join(sorted(group["resolution"].astype(str).unique())),
        "context_hours": ",".join(map(str, sorted(group["context_hours"].unique()))),
        "context_days_equivalent": ",".join(map(str, sorted(group["context_days_equivalent"].unique()))),
        "prediction_hours": ",".join(map(str, sorted(group["prediction_hours"].unique()))),
        "prediction_steps": ",".join(map(str, sorted(group["prediction_steps"].unique()))),
        "MAE": float(np.mean(absolute_error)),
        "RMSE": rmse,
        "R2": float(1 - residual_sum_of_squares / total_sum_of_squares) if total_sum_of_squares > 0 else np.nan,
        "CVRMSE_percent": float(rmse / actual_mean * 100) if actual_mean != 0 else np.nan,
        "sMAPE_percent": float(np.nanmean(group["sape"]) * 100),
        "mean_error": float(np.mean(error)),
        "median_absolute_error": float(np.median(absolute_error)),
        "max_absolute_error": float(np.max(absolute_error)),
        "prediction_seconds_total": float(group.groupby("forecast_start", sort=False)["prediction_seconds"].first().sum()),
        "prediction_seconds_mean_per_forecast_start": float(group.groupby("forecast_start", sort=False)["prediction_seconds"].first().mean()),
    }
    if "forecast_loop_seconds" in group.columns:
        per_forecast_loop = group.groupby("forecast_start", sort=False)["forecast_loop_seconds"].first()
        metrics["forecast_loop_seconds_total"] = float(per_forecast_loop.sum())
        metrics["forecast_loop_seconds_mean_per_forecast_start"] = float(per_forecast_loop.mean())
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


def wandb_daily_metrics(metrics: dict[str, object]) -> dict[str, float]:
    keys = [
        "MAE",
        "RMSE",
        "CVRMSE_percent",
        "sMAPE_percent",
        "mean_error",
        "median_absolute_error",
        "max_absolute_error",
        "coverage_68_percent",
        "mean_width_68",
        "prediction_seconds",
    ]
    return {f"daily/{key}": float(metrics[key]) for key in keys if key in metrics and pd.notna(metrics[key])}


def log_wandb_summary(wandb_run, summary: pd.DataFrame) -> None:
    if wandb_run is None:
        return
    for _, row in summary.iterrows():
        scope = safe_name(str(row["metric_scope"]))
        payload = {}
        for key in [
            "MAE",
            "RMSE",
            "CVRMSE_percent",
            "sMAPE_percent",
            "coverage_68_percent",
            "mean_width_68",
            "prediction_seconds_mean_excluding_first",
            "estimated_setup_time",
        ]:
            if key in row and pd.notna(row[key]):
                payload[f"summary/{scope}/{key}"] = float(row[key])
        if payload:
            wandb_run.log(payload)


def predict_one_day(
    pipeline,
    data: pd.DataFrame,
    selection: str,
    week_id: str,
    forecast_start: pd.Timestamp,
    args: argparse.Namespace,
    data_loading_seconds: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    forecast_loop_start = time.perf_counter()
    context_df, future_df, test_df, step = build_windows(
        data,
        forecast_start=forecast_start,
        context_hours=args.context_hours,
        prediction_hours=args.prediction_hours,
        resolution=args.resolution,
    )
    validate_model_inputs(context_df, future_df)

    if getattr(pipeline, "max_context_length", 0) < len(context_df):
        raise ValueError(
            "TabPFN-TS would truncate the context: "
            f"max_context_length={pipeline.max_context_length}, context rows={len(context_df)}."
        )

    prediction_start = time.perf_counter()
    pred_df = pipeline.predict_df(
        context_df=context_df,
        future_df=future_df,
        quantiles=DEFAULT_QUANTILES,
    )
    prediction_seconds = time.perf_counter() - prediction_start

    forecast = build_forecast_output(pred_df, test_df, step, DEFAULT_QUANTILES)
    forecast.insert(0, "selection", selection)
    forecast.insert(1, "week_id", week_id)
    forecast.insert(2, "forecast_start", forecast_start.tz_localize(None))
    forecast.insert(3, "context_start", (forecast_start - pd.Timedelta(hours=args.context_hours)).tz_localize(None))
    forecast.insert(4, "context_hours", args.context_hours)
    forecast.insert(5, "context_days_equivalent", args.context_hours / 24)
    forecast.insert(6, "resolution", args.resolution)
    forecast.insert(7, "prediction_hours", args.prediction_hours)
    forecast.insert(8, "prediction_steps", prediction_steps(args.prediction_hours, args.resolution))
    forecast.insert(9, "mode", args.mode)
    forecast.insert(10, "prediction_seconds", prediction_seconds)
    forecast.insert(11, "context_rows", len(context_df))

    metrics = calculate_forecast_metrics(
        forecast=forecast,
        args=args,
        selection=selection,
        week_id=week_id,
        forecast_start=forecast_start,
        data_loading_seconds=data_loading_seconds,
        prediction_seconds=prediction_seconds,
        n_context_rows=len(context_df),
    )
    forecast_loop_seconds = time.perf_counter() - forecast_loop_start
    forecast["forecast_loop_seconds"] = forecast_loop_seconds
    metrics["forecast_loop_seconds"] = forecast_loop_seconds
    return forecast, metrics


def selected_weeks_source_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Selected weeks file not found: {path}")
    return {"path": str(path), "exists": True, "sha256": sha256_file(path)}


def build_metadata(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    selected_weeks: pd.DataFrame,
    total_seconds: float,
    wandb_run,
) -> dict[str, object]:
    metadata_payload = metadata_envelope(
        run_id=run_id,
        run_name=args.run_name,
        run_dir=run_dir,
        script_path=__file__,
        packages=TABPFN_PACKAGES,
    )
    metadata_payload.update({
        "args": {
            "resolution": args.resolution,
            "context_hours": args.context_hours,
            "context_days_equivalent": args.context_hours / 24,
            "prediction_hours": args.prediction_hours,
            "prediction_steps": prediction_steps(args.prediction_hours, args.resolution),
            "mode": args.mode,
            "output_dir": str(args.output_dir),
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_enabled": not args.disable_wandb,
        },
        "selected_forecast_weeks": selected_weeks.to_dict(orient="records"),
        "selected_weeks_source": selected_weeks_source_metadata(args.selected_weeks_path),
        "data": {
            "heat_path": str(args.heat_path),
            "heat_sha256": sha256_file(args.heat_path),
            "weather_path": str(args.weather_path),
            "weather_sha256": sha256_file(args.weather_path),
            "weather_columns": WEATHER_COLUMNS,
        },
        "time_handling": {
            "timezone": TIMEZONE,
            "forecast_starts_are_local_midnight": True,
            "context_window_unit": "hours",
        },
        "forecast_windowing": {
            "mode": "non_overlapping_within_selected_week",
            "window_hours": args.prediction_hours,
            "step_hours": args.prediction_hours,
            "week_duration_hours": 168,
            "incomplete_final_windows": "discarded",
        },
        "resolution_handling": {
            "resolution": args.resolution,
            "frequency": RESOLUTION_TO_FREQ[args.resolution],
            "steps_per_hour": STEPS_PER_HOUR_BY_RESOLUTION[args.resolution],
            "hourly_aggregation": "mean of complete hours using inferred source cadence",
        },
        "quantiles": DEFAULT_QUANTILES,
        "interval_definitions": INTERVAL_DEFINITIONS,
        "wandb": {
            "run_id": getattr(wandb_run, "id", None) if wandb_run is not None else None,
            "run_name": getattr(wandb_run, "name", None) if wandb_run is not None else None,
        },
        "total_seconds": total_seconds,
    })
    return metadata_payload


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()
    selected_weeks = load_selected_weeks(args.selected_weeks_path)
    starts = forecast_starts_for_selected_weeks(
        selected_weeks,
        prediction_hours=args.prediction_hours,
        max_forecast_starts=args.max_forecast_starts,
    )

    print(f"Forecast starts requested: {len(starts):,}")
    print("Forecast windowing: non-overlapping within each selected week; incomplete final windows discarded")
    print(f"Context hours requested: {args.context_hours}")
    print(f"Context days equivalent: {args.context_hours / 24:g}")
    print(f"Resolution: {args.resolution}")
    print(f"Prediction hours: {args.prediction_hours}")
    print(f"Prediction steps: {prediction_steps(args.prediction_hours, args.resolution)}")
    print(f"Quantiles saved: {', '.join(quantile_column_name(q) for q in DEFAULT_QUANTILES)}")

    load_start = time.perf_counter()
    data = load_data(args.heat_path, args.weather_path, resolution=args.resolution)
    data_loading_seconds = time.perf_counter() - load_start
    validate_complete_year(
        data,
        year=2024,
        step=EXPECTED_STEP_BY_RESOLUTION[args.resolution],
        value_columns=["heat", *WEATHER_COLUMNS],
        label="selected-week forecast data",
    )

    context_lengths = []
    for selection, week_id, forecast_start in starts:
        context_df, future_df, _, _ = build_windows(
            data,
            forecast_start=forecast_start,
            context_hours=args.context_hours,
            prediction_hours=args.prediction_hours,
            resolution=args.resolution,
        )
        validate_model_inputs(context_df, future_df)
        context_lengths.append(len(context_df))
        if args.dry_run:
            print(
                f"Validated {selection} {week_id} {forecast_start}: "
                f"context_rows={len(context_df):,}, future_rows={len(future_df):,}"
            )
    max_context_length = max(context_lengths)
    print(f"TabPFN-TS max_context_length set to: {max_context_length:,}")
    if args.dry_run:
        print("Dry run complete. No model loaded and no outputs written.")
        return

    run_name_parts = [
        args.resolution,
        f"context{args.context_hours}h",
        f"horizon{args.prediction_hours}h",
        args.mode.lower(),
    ]
    if args.run_name:
        run_name_parts.append(safe_name(args.run_name))
    run_id = make_run_id(args.run_id, "_".join(run_name_parts))
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
    print(f"Run directory: {run_dir}")

    pipeline = initialize_pipeline(args.mode, max_context_length=max_context_length)
    wandb_run = initialize_wandb(
        args,
        run_id,
        {
            "run_id": run_id,
            "run_name": args.run_name,
            "resolution": args.resolution,
            "resolution_frequency": RESOLUTION_TO_FREQ[args.resolution],
            "context_hours": args.context_hours,
            "context_days_equivalent": args.context_hours / 24,
            "prediction_hours": args.prediction_hours,
            "prediction_steps": prediction_steps(args.prediction_hours, args.resolution),
            "mode": args.mode,
            "heat_path": str(args.heat_path),
            "weather_path": str(args.weather_path),
            "weather_columns": WEATHER_COLUMNS,
            "selected_weeks_path": str(args.selected_weeks_path),
            "selected_forecast_weeks": selected_weeks.to_dict(orient="records"),
            "quantiles": DEFAULT_QUANTILES,
        },
    )

    raw_forecasts = []
    metric_rows = []
    for day_number, (selection, week_id, forecast_start) in enumerate(starts, start=1):
        print(f"[{day_number}/{len(starts)}] Predicting {selection} {week_id}: {forecast_start}")
        forecast, metrics = predict_one_day(
            pipeline=pipeline,
            data=data,
            selection=selection,
            week_id=week_id,
            forecast_start=forecast_start,
            args=args,
            data_loading_seconds=data_loading_seconds,
        )
        raw_forecasts.append(forecast)
        metric_rows.append(metrics)
        if wandb_run is not None:
            wandb_run.log(wandb_daily_metrics(metrics), step=day_number)
        print(
            f"[{day_number}/{len(starts)}] MAE={metrics['MAE']:.3f}, "
            f"RMSE={metrics['RMSE']:.3f}, "
            f"coverage68={metrics.get('coverage_68_percent', np.nan):.1f}%, "
            f"prediction={metrics['prediction_seconds']:.2f}s"
        )

    raw = pd.concat(raw_forecasts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = calculate_summary_metrics(raw)
    raw.to_csv(raw_output_path, index=False)
    metrics.to_csv(metrics_output_path, index=False)
    summary.to_csv(summary_output_path, index=False)

    if wandb_run is not None:
        log_wandb_summary(wandb_run, summary)
        wandb_run.finish()

    total_seconds = time.perf_counter() - total_start
    write_metadata(
        metadata_path,
        build_metadata(args, run_id, run_dir, selected_weeks, total_seconds, wandb_run),
    )

    print(f"Completed forecast starts: {len(starts):,}")
    print(f"Saved raw predictions: {raw_output_path}")
    print(f"Saved per-start metrics: {metrics_output_path}")
    print(f"Saved summary metrics: {summary_output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Total seconds: {total_seconds:.2f}")


if __name__ == "__main__":
    main()
