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
    forecast_starts_for_selected_weeks,
    load_selected_weeks,
    safe_name,
    sha256_file,
    validate_complete_year,
)
from tabpfn_ts_heat_backtest_extreme_weeks_2024 import (
    DEFAULT_QUANTILES,
    INTERVAL_DEFINITIONS,
    add_interval_metrics,
    build_forecast_output,
    pinball_loss,
    quantile_column_name,
    selected_weeks_source_metadata,
)
from utils import TABPFN_PACKAGES, initialize_wandb, make_run_id, metadata_envelope, write_metadata
from tabpfn_ts_heat_forecast import (
    DEFAULT_HEAT_PATH,
    RESOLUTION_TO_FREQ,
    STEPS_PER_HOUR_BY_RESOLUTION,
    WEATHER_COLUMNS,
    load_data,
    make_tabpfn_frame,
    prediction_steps,
    validate_model_inputs,
)


DEFAULT_WEATHER_PATH = Path("flensburg/weather/flensburg_weather_temperature.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/relevant_context_comparison")
DEFAULT_RUN_NAME = "tabpfn_relevant_context_comparison"
DEFAULT_WANDB_PROJECT = "timeseries-forecasting"
RESOLUTION = "hourly"
PREDICTION_HOURS = 24
EXPECTED_CONTEXT_HOURS = 12 * 7 * 24
RECENT_CONTEXT_HOURS = 6 * 7 * 24
LAST_YEAR_HALF_WINDOW_HOURS = 3 * 7 * 24
CONTEXT_STRATEGIES = ("recent_12w", "recent_6w_plus_last_year_6w")
EXPERIMENT_FAMILY = "relevant_context_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare recent-only and relevant-last-year context strategies for "
            "hourly TabPFN-TS forecasts on signal-error-free 2024 extreme weeks."
        )
    )
    parser.add_argument("--heat-path", type=Path, default=DEFAULT_HEAT_PATH)
    parser.add_argument("--weather-path", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--selected-weeks-path", type=Path, default=DEFAULT_SELECTED_WEEKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--mode", choices=("CLIENT", "LOCAL"), default="LOCAL")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--max-forecast-starts", type=int, default=None)
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Run without logging scalar metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all context/future windows without running TabPFN-TS or writing outputs.",
    )
    return parser.parse_args()


def initialize_pipeline(mode: str, max_context_length: int):
    try:
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
        from tabpfn_time_series.features import CalendarFeature
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    return TabPFNTSPipeline(
        tabpfn_mode=getattr(TabPFNMode, mode),
        max_context_length=max_context_length,
        temporal_features=[CalendarFeature()],
    )


def select_interval(
    data: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    expected_rows: int,
    label: str,
) -> pd.DataFrame:
    block = data[(data["timestamp"] >= start) & (data["timestamp"] < end_exclusive)].copy()
    if len(block) != expected_rows:
        raise ValueError(
            f"{label} has {len(block)} rows, expected {expected_rows}. "
            f"Window: [{start}, {end_exclusive})."
        )
    return block


def validate_context_raw(context_raw: pd.DataFrame, strategy: str, forecast_start: pd.Timestamp) -> None:
    if len(context_raw) != EXPECTED_CONTEXT_HOURS:
        raise ValueError(
            f"{strategy} context for {forecast_start} has {len(context_raw)} rows, "
            f"expected {EXPECTED_CONTEXT_HOURS}."
        )
    if context_raw["timestamp"].duplicated().any():
        duplicated = context_raw.loc[context_raw["timestamp"].duplicated(), "timestamp"].head().tolist()
        raise ValueError(f"{strategy} context contains duplicate timestamps: {duplicated}")
    if not (context_raw["timestamp"] < forecast_start).all():
        bad = context_raw.loc[context_raw["timestamp"] >= forecast_start, "timestamp"].head().tolist()
        raise ValueError(f"{strategy} context includes future/current timestamps: {bad}")
    required_columns = ["heat", *WEATHER_COLUMNS]
    if context_raw[required_columns].isna().any().any():
        missing_counts = context_raw[required_columns].isna().sum().to_dict()
        raise ValueError(f"{strategy} context has missing values: {missing_counts}")


def context_for_strategy(
    data: pd.DataFrame,
    strategy: str,
    selection: str,
    week_id: str,
    forecast_start: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    if strategy == "recent_12w":
        block_start = forecast_start - pd.Timedelta(hours=EXPECTED_CONTEXT_HOURS)
        block_end = forecast_start
        context_raw = select_interval(
            data,
            block_start,
            block_end,
            EXPECTED_CONTEXT_HOURS,
            f"{strategy}/recent_12w",
        )
        rows.append(
            {
                "context_strategy": strategy,
                "selection": selection,
                "week_id": week_id,
                "forecast_start": forecast_start.tz_localize(None).isoformat(),
                "block_name": "recent_12w",
                "block_start": block_start.tz_localize(None).isoformat(),
                "block_end_exclusive": block_end.tz_localize(None).isoformat(),
                "last_year_anchor": None,
                "n_rows": len(context_raw),
            }
        )
    elif strategy == "recent_6w_plus_last_year_6w":
        recent_start = forecast_start - pd.Timedelta(hours=RECENT_CONTEXT_HOURS)
        recent_end = forecast_start
        last_year_anchor = forecast_start - pd.DateOffset(years=1)
        last_year_start = last_year_anchor - pd.Timedelta(hours=LAST_YEAR_HALF_WINDOW_HOURS)
        last_year_end = last_year_anchor + pd.Timedelta(hours=LAST_YEAR_HALF_WINDOW_HOURS)
        recent_raw = select_interval(
            data,
            recent_start,
            recent_end,
            RECENT_CONTEXT_HOURS,
            f"{strategy}/recent_6w",
        )
        last_year_raw = select_interval(
            data,
            last_year_start,
            last_year_end,
            RECENT_CONTEXT_HOURS,
            f"{strategy}/last_year_6w",
        )
        context_raw = pd.concat([last_year_raw, recent_raw], ignore_index=True)
        for block_name, block_start, block_end, block_rows in (
            ("last_year_6w", last_year_start, last_year_end, len(last_year_raw)),
            ("recent_6w", recent_start, recent_end, len(recent_raw)),
        ):
            rows.append(
                {
                    "context_strategy": strategy,
                    "selection": selection,
                    "week_id": week_id,
                    "forecast_start": forecast_start.tz_localize(None).isoformat(),
                    "block_name": block_name,
                    "block_start": block_start.tz_localize(None).isoformat(),
                    "block_end_exclusive": block_end.tz_localize(None).isoformat(),
                    "last_year_anchor": last_year_anchor.tz_localize(None).isoformat(),
                    "n_rows": block_rows,
                }
            )
    else:
        raise ValueError(f"Unknown context strategy: {strategy}")

    context_raw = context_raw.sort_values("timestamp").reset_index(drop=True)
    validate_context_raw(context_raw, strategy, forecast_start)
    return make_tabpfn_frame(context_raw), rows


def future_for_forecast(
    data: pd.DataFrame,
    forecast_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast_end = forecast_start + pd.Timedelta(hours=PREDICTION_HOURS)
    future_raw = data[(data["timestamp"] >= forecast_start) & (data["timestamp"] < forecast_end)].copy()
    expected_rows = prediction_steps(PREDICTION_HOURS, RESOLUTION)
    if len(future_raw) != expected_rows:
        raise ValueError(
            f"Future horizon for {forecast_start} has {len(future_raw)} rows, expected {expected_rows}."
        )
    if future_raw[WEATHER_COLUMNS].isna().any().any():
        missing_counts = future_raw[WEATHER_COLUMNS].isna().sum().to_dict()
        raise ValueError(f"Future covariates contain missing values: {missing_counts}")

    future_df = make_tabpfn_frame(future_raw).drop(columns=["target"])
    if "target" in future_df.columns:
        raise ValueError("future_df must not contain target.")
    test_df = future_raw.rename(columns={"heat": "actual_heat"})[
        ["timestamp", "actual_heat", *WEATHER_COLUMNS]
    ].copy()
    test_df["timestamp"] = test_df["timestamp"].dt.tz_localize(None)
    return future_df, test_df


def add_prediction_metadata(
    forecast: pd.DataFrame,
    strategy: str,
    selection: str,
    week_id: str,
    forecast_start: pd.Timestamp,
    mode: str,
    prediction_seconds: float,
    context_rows: int,
) -> pd.DataFrame:
    forecast = forecast.copy()
    forecast.insert(0, "context_strategy", strategy)
    forecast.insert(1, "selection", selection)
    forecast.insert(2, "week_id", week_id)
    forecast.insert(3, "forecast_start", forecast_start.tz_localize(None).isoformat())
    forecast.insert(4, "context_hours", EXPECTED_CONTEXT_HOURS)
    forecast.insert(5, "context_days_equivalent", EXPECTED_CONTEXT_HOURS / 24)
    forecast.insert(6, "resolution", RESOLUTION)
    forecast.insert(7, "prediction_hours", PREDICTION_HOURS)
    forecast.insert(8, "prediction_steps", prediction_steps(PREDICTION_HOURS, RESOLUTION))
    forecast.insert(9, "mode", mode)
    forecast.insert(10, "prediction_seconds", prediction_seconds)
    forecast.insert(11, "context_rows", context_rows)
    return forecast


def calculate_forecast_metrics(
    forecast: pd.DataFrame,
    strategy: str,
    selection: str,
    week_id: str,
    forecast_start: pd.Timestamp,
    data_loading_seconds: float,
    prediction_seconds: float,
    mode: str,
    n_context_rows: int,
) -> dict[str, object]:
    error = forecast["error"].to_numpy(dtype=float)
    actual = forecast["actual_heat"].to_numpy(dtype=float)
    absolute_error = np.abs(error)
    rmse = math.sqrt(float(np.mean(error**2)))
    total_sum_of_squares = float(np.sum((actual - np.mean(actual)) ** 2))
    residual_sum_of_squares = float(np.sum(error**2))
    mean_actual = float(np.mean(actual))
    metrics: dict[str, object] = {
        "context_strategy": strategy,
        "selection": selection,
        "week_id": week_id,
        "forecast_start": forecast_start.tz_localize(None).isoformat(),
        "resolution": RESOLUTION,
        "context_hours": EXPECTED_CONTEXT_HOURS,
        "context_days_equivalent": EXPECTED_CONTEXT_HOURS / 24,
        "prediction_hours": PREDICTION_HOURS,
        "prediction_steps": prediction_steps(PREDICTION_HOURS, RESOLUTION),
        "mode": mode,
        "n_context_rows": n_context_rows,
        "n_forecast_rows": len(forecast),
        "MAE": float(np.mean(absolute_error)),
        "RMSE": rmse,
        "R2": float(1 - residual_sum_of_squares / total_sum_of_squares)
        if total_sum_of_squares > 0
        else np.nan,
        "CVRMSE_percent": float(rmse / mean_actual * 100) if mean_actual != 0 else np.nan,
        "sMAPE_percent": float(np.nanmean(forecast["sape"]) * 100),
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
            metrics[f"pinball_loss_{column}"] = pinball_loss(
                actual,
                forecast[column].to_numpy(dtype=float),
                quantile,
            )
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


def calculate_group_metrics(strategy: str, scope: str, group: pd.DataFrame) -> dict[str, object]:
    error = group["error"].to_numpy(dtype=float)
    actual = group["actual_heat"].to_numpy(dtype=float)
    absolute_error = np.abs(error)
    rmse = math.sqrt(float(np.mean(error**2)))
    actual_mean = float(np.mean(actual))
    total_sum_of_squares = float(np.sum((actual - actual_mean) ** 2))
    residual_sum_of_squares = float(np.sum(error**2))
    metrics: dict[str, object] = {
        "context_strategy": strategy,
        "metric_scope": scope,
        "n_forecast_starts": int(group["forecast_start"].nunique()),
        "n_rows": int(len(group)),
        "resolution": RESOLUTION,
        "context_hours": EXPECTED_CONTEXT_HOURS,
        "context_days_equivalent": EXPECTED_CONTEXT_HOURS / 24,
        "prediction_hours": PREDICTION_HOURS,
        "prediction_steps": prediction_steps(PREDICTION_HOURS, RESOLUTION),
        "MAE": float(np.mean(absolute_error)),
        "RMSE": rmse,
        "R2": float(1 - residual_sum_of_squares / total_sum_of_squares)
        if total_sum_of_squares > 0
        else np.nan,
        "CVRMSE_percent": float(rmse / actual_mean * 100) if actual_mean != 0 else np.nan,
        "sMAPE_percent": float(np.nanmean(group["sape"]) * 100),
        "mean_error": float(np.mean(error)),
        "median_absolute_error": float(np.median(absolute_error)),
        "max_absolute_error": float(np.max(absolute_error)),
        "prediction_seconds_total": float(
            group.groupby("forecast_start", sort=False)["prediction_seconds"].first().sum()
        ),
        "prediction_seconds_mean_per_forecast_start": float(
            group.groupby("forecast_start", sort=False)["prediction_seconds"].first().mean()
        ),
    }
    add_interval_metrics(metrics, group)
    if scope == "all":
        add_run_timing_metrics(metrics, group)
    return metrics


def calculate_summary_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, strategy_group in raw.groupby("context_strategy", sort=False):
        rows.append(calculate_group_metrics(strategy, "all", strategy_group))
        rows.extend(
            calculate_group_metrics(strategy, f"selection:{selection}", group)
            for selection, group in strategy_group.groupby("selection", sort=True)
        )
        rows.extend(
            calculate_group_metrics(strategy, f"week:{week_id}", group)
            for week_id, group in strategy_group.groupby("week_id", sort=True)
        )
    return pd.DataFrame(rows)


def log_wandb_summary(wandb_run, summary: pd.DataFrame) -> None:
    if wandb_run is None:
        return
    payload = {}
    for _, row in summary.iterrows():
        strategy = safe_name(str(row["context_strategy"]))
        scope = safe_name(str(row["metric_scope"]))
        for key in [
            "MAE",
            "RMSE",
            "R2",
            "CVRMSE_percent",
            "sMAPE_percent",
            "coverage_68_percent",
            "mean_width_68",
            "prediction_seconds_mean_excluding_first",
            "estimated_setup_time",
        ]:
            if key in row and pd.notna(row[key]):
                payload[f"summary/{strategy}/{scope}/{key}"] = float(row[key])
    if payload:
        wandb_run.log(payload)


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
        "experiment_family": EXPERIMENT_FAMILY,
        "args": {
            "resolution": RESOLUTION,
            "context_hours": EXPECTED_CONTEXT_HOURS,
            "context_days_equivalent": EXPECTED_CONTEXT_HOURS / 24,
            "prediction_hours": PREDICTION_HOURS,
            "prediction_steps": prediction_steps(PREDICTION_HOURS, RESOLUTION),
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
        "context_strategies": {
            "recent_12w": {
                "description": "12 weeks immediately before each forecast start",
                "context_hours": EXPECTED_CONTEXT_HOURS,
            },
            "recent_6w_plus_last_year_6w": {
                "description": (
                    "6 weeks immediately before each forecast start plus 6 weeks centered "
                    "on the same calendar time in the previous year"
                ),
                "recent_context_hours": RECENT_CONTEXT_HOURS,
                "last_year_context_hours": RECENT_CONTEXT_HOURS,
                "last_year_half_window_hours": LAST_YEAR_HALF_WINDOW_HOURS,
            },
        },
        "temporal_features": {
            "enabled": ["CalendarFeature"],
            "running_index_feature": False,
            "auto_seasonal_feature": False,
        },
        "forecast_windowing": {
            "mode": "non_overlapping_within_selected_week",
            "window_hours": PREDICTION_HOURS,
            "step_hours": PREDICTION_HOURS,
            "week_duration_hours": 168,
            "incomplete_final_windows": "discarded",
        },
        "resolution_handling": {
            "resolution": RESOLUTION,
            "frequency": RESOLUTION_TO_FREQ[RESOLUTION],
            "steps_per_hour": STEPS_PER_HOUR_BY_RESOLUTION[RESOLUTION],
            "hourly_aggregation": (
                "cadence-aware: native hourly data is retained; finer-resolution data is averaged "
                "over complete hours using the inferred source cadence"
            ),
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


def validate_windows(
    data: pd.DataFrame,
    starts: list[tuple[str, str, pd.Timestamp]],
) -> pd.DataFrame:
    rows = []
    for selection, week_id, forecast_start in starts:
        future_df, _ = future_for_forecast(data, forecast_start)
        if "target" in future_df.columns:
            raise ValueError("future_df must never contain target.")
        for strategy in CONTEXT_STRATEGIES:
            context_df, context_rows = context_for_strategy(data, strategy, selection, week_id, forecast_start)
            if len(context_df) != EXPECTED_CONTEXT_HOURS:
                raise ValueError(f"{strategy} produced {len(context_df)} context rows.")
            rows.extend(context_rows)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()

    selected_weeks = load_selected_weeks(args.selected_weeks_path)
    starts = forecast_starts_for_selected_weeks(
        selected_weeks,
        prediction_hours=PREDICTION_HOURS,
        max_forecast_starts=args.max_forecast_starts,
    )
    if not starts:
        raise ValueError("No forecast starts selected.")

    print(f"Forecast starts requested: {len(starts):,}")
    print(f"Context strategies: {', '.join(CONTEXT_STRATEGIES)}")
    print(f"Context rows per strategy/start: {EXPECTED_CONTEXT_HOURS:,}")
    print("Temporal features: CalendarFeature only")
    print(f"Weather columns: {', '.join(WEATHER_COLUMNS)}")

    load_start = time.perf_counter()
    data = load_data(args.heat_path, args.weather_path, resolution=RESOLUTION)
    data_loading_seconds = time.perf_counter() - load_start
    validate_complete_year(
        data,
        year=2024,
        step=HOURLY_STEP,
        value_columns=["heat", *WEATHER_COLUMNS],
        label="selected-week forecast data",
    )
    context_windows = validate_windows(data, starts)

    expected_context_window_rows = len(starts) * 3
    if len(context_windows) != expected_context_window_rows:
        raise ValueError(
            f"Expected {expected_context_window_rows} context-window rows, got {len(context_windows)}."
        )

    if args.dry_run:
        print("Dry run complete. No predictions or outputs written.")
        print(f"Validated forecast starts: {len(starts)}")
        print(f"Validated strategy evaluations: {len(starts) * len(CONTEXT_STRATEGIES)}")
        print(f"Validated context-window rows: {len(context_windows)}")
        return

    run_id = make_run_id(args.run_id, args.run_name)
    run_dir = args.output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    raw_output_path = run_dir / "raw_predictions.csv"
    metrics_output_path = run_dir / "metrics_per_forecast_start.csv"
    summary_output_path = run_dir / "metrics_summary.csv"
    metadata_path = run_dir / "run_metadata.json"
    command_path = run_dir / "command.txt"
    context_windows_path = run_dir / "context_windows.csv"
    command_path.write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    wandb_run = initialize_wandb(
        args,
        run_id,
        {
            "run_id": run_id,
            "run_name": args.run_name,
            "experiment_family": EXPERIMENT_FAMILY,
            "resolution": RESOLUTION,
            "resolution_frequency": RESOLUTION_TO_FREQ[RESOLUTION],
            "prediction_hours": PREDICTION_HOURS,
            "prediction_steps": prediction_steps(PREDICTION_HOURS, RESOLUTION),
            "mode": args.mode,
            "heat_path": str(args.heat_path),
            "weather_path": str(args.weather_path),
            "weather_columns": WEATHER_COLUMNS,
            "selected_weeks_path": str(args.selected_weeks_path),
            "selected_forecast_weeks": selected_weeks.to_dict(orient="records"),
            "context_strategies": CONTEXT_STRATEGIES,
            "recent_12w_context_hours": EXPECTED_CONTEXT_HOURS,
            "mixed_recent_context_hours": RECENT_CONTEXT_HOURS,
            "mixed_last_year_context_hours": RECENT_CONTEXT_HOURS,
            "temporal_features": ["CalendarFeature"],
            "running_index_feature": False,
            "auto_seasonal_feature": False,
            "quantiles": DEFAULT_QUANTILES,
        },
    )
    pipeline = initialize_pipeline(args.mode, max_context_length=EXPECTED_CONTEXT_HOURS)
    if getattr(pipeline, "max_context_length", 0) < EXPECTED_CONTEXT_HOURS:
        raise ValueError(
            "TabPFN-TS would truncate the context: "
            f"max_context_length={pipeline.max_context_length}, context rows={EXPECTED_CONTEXT_HOURS}."
        )

    raw_rows = []
    metric_rows = []
    total_evaluations = len(starts) * len(CONTEXT_STRATEGIES)
    evaluation_number = 0
    for selection, week_id, forecast_start in starts:
        future_df, test_df = future_for_forecast(data, forecast_start)
        for strategy in CONTEXT_STRATEGIES:
            evaluation_number += 1
            print(
                f"[{evaluation_number}/{total_evaluations}] "
                f"{strategy} {selection} {week_id} {forecast_start}"
            )
            context_df, _ = context_for_strategy(data, strategy, selection, week_id, forecast_start)
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
            forecast = build_forecast_output(
                pred_df,
                test_df,
                pd.Timedelta(hours=1),
                DEFAULT_QUANTILES,
            )
            forecast = add_prediction_metadata(
                forecast=forecast,
                strategy=strategy,
                selection=selection,
                week_id=week_id,
                forecast_start=forecast_start,
                mode=args.mode,
                prediction_seconds=prediction_seconds,
                context_rows=len(context_df),
            )
            metrics = calculate_forecast_metrics(
                forecast=forecast,
                strategy=strategy,
                selection=selection,
                week_id=week_id,
                forecast_start=forecast_start,
                data_loading_seconds=data_loading_seconds,
                prediction_seconds=prediction_seconds,
                mode=args.mode,
                n_context_rows=len(context_df),
            )
            raw_rows.append(forecast)
            metric_rows.append(metrics)
            print(
                f"  RMSE={metrics['RMSE']:.3f}, "
                f"CVRMSE={metrics['CVRMSE_percent']:.2f}%, "
                f"prediction={prediction_seconds:.2f}s"
            )

    raw = pd.concat(raw_rows, ignore_index=True)
    per_start_metrics = pd.DataFrame(metric_rows)
    summary = calculate_summary_metrics(raw)

    raw.to_csv(raw_output_path, index=False)
    per_start_metrics.to_csv(metrics_output_path, index=False)
    summary.to_csv(summary_output_path, index=False)
    context_windows.to_csv(context_windows_path, index=False)
    log_wandb_summary(wandb_run, summary)

    total_seconds = time.perf_counter() - total_start
    write_metadata(
        metadata_path,
        build_metadata(
            args=args,
            run_id=run_id,
            run_dir=run_dir,
            selected_weeks=selected_weeks,
            total_seconds=total_seconds,
            wandb_run=wandb_run,
        ),
    )

    print(f"Saved raw predictions: {raw_output_path}")
    print(f"Saved per-start metrics: {metrics_output_path}")
    print(f"Saved summary metrics: {summary_output_path}")
    print(f"Saved context windows: {context_windows_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Total seconds: {total_seconds:.2f}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
