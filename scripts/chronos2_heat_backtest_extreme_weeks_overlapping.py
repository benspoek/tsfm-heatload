from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import chronos2_heat_backtest_extreme_weeks_2024 as chronos_extreme
from full_year_forecasting_utils import (
    DEFAULT_SELECTED_WEEKS_PATH,
    UnsupportedDataCadenceError,
    add_synthetic_model_clock,
    forecast_starts_for_selected_weeks,
    load_merged_data,
    load_selected_weeks,
    local_midnight,
    validate_complete_year,
)
from utils import (
    CHRONOS_PACKAGES,
    initialize_wandb,
    make_run_id,
    metadata_envelope,
    parse_weather_columns,
    write_metadata,
)


DEFAULT_HEAT_PATH = chronos_extreme.DEFAULT_HEAT_PATH
DEFAULT_WEATHER_PATH = chronos_extreme.DEFAULT_WEATHER_PATH
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/extreme_weeks_2024")
DEFAULT_WANDB_PROJECT = "timeseries-forecasting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run overlapping direct Chronos-2 forecasts on the selected extreme weeks "
            "and evaluate them as a latest-forecast-wins operational trajectory."
        )
    )
    parser.add_argument("--heat-path", type=Path, default=DEFAULT_HEAT_PATH)
    parser.add_argument("--weather-path", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--selected-weeks-path", type=Path, default=DEFAULT_SELECTED_WEEKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resolution", choices=tuple(chronos_extreme.RESOLUTION_TO_STEP), default="hourly")
    parser.add_argument("--context-hours", type=int, default=2016)
    parser.add_argument("--prediction-hours", type=int, default=24)
    parser.add_argument("--forecast-every-hours", type=int, default=4)
    parser.add_argument("--weather-columns", default="temperature")
    parser.add_argument("--model-path", default=chronos_extreme.DEFAULT_MODEL_PATH)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--max-context-steps", type=int, default=chronos_extreme.DEFAULT_MAX_CONTEXT_STEPS)
    parser.add_argument("--max-forecast-starts", type=int, default=None)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def forecast_starts_for_weeks(
    selected_weeks: pd.DataFrame,
    forecast_every_hours: int,
) -> list[tuple[str, str, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    issued_starts = forecast_starts_for_selected_weeks(
        selected_weeks,
        prediction_hours=1,
        stride_hours=forecast_every_hours,
        allow_horizon_past_week=True,
    )
    week_starts = {
        str(row.week_id): local_midnight(row.week_start)
        for row in selected_weeks.itertuples(index=False)
    }
    return [
        (
            selection,
            week_id,
            week_starts[week_id],
            week_starts[week_id] + pd.DateOffset(days=7),
            forecast_start,
        )
        for selection, week_id, forecast_start in issued_starts
    ]


def add_week_bounds(forecast_df: pd.DataFrame, week_start: pd.Timestamp, week_end: pd.Timestamp) -> pd.DataFrame:
    out = forecast_df.copy()
    out.insert(3, "week_start", week_start.isoformat())
    out.insert(4, "week_end", week_end.isoformat())
    out["within_selected_week"] = (out["timestamp"] >= week_start) & (out["timestamp"] < week_end)
    return out


def latest_forecast_wins(raw: pd.DataFrame) -> pd.DataFrame:
    within = raw[raw["within_selected_week"]].copy()
    if within.empty:
        raise ValueError("No issued forecast rows fall inside the selected weeks.")
    within = within.sort_values(["selection", "week_id", "timestamp", "forecast_start"]).reset_index(drop=True)
    return within.drop_duplicates(["selection", "week_id", "timestamp"], keep="last").reset_index(drop=True)


def calculate_per_forecast_start_metrics(operational: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["selection", "week_id", "forecast_start"]
    for (selection, week_id, forecast_start), group in operational.groupby(group_columns, sort=True):
        row = chronos_extreme.calculate_group_metrics("operational_forecast_start", group)
        row.update(
            {
                "selection": selection,
                "week_id": week_id,
                "forecast_start": forecast_start,
                "n_forecast_rows": len(group),
                "n_context_rows": int(group["effective_context_steps"].iloc[0]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_metadata(
    args: argparse.Namespace,
    run_id: str,
    selected_weeks: pd.DataFrame,
    starts: list[tuple[str, str, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    weather_columns: list[str],
    total_seconds: float,
    wandb_run,
) -> dict[str, object]:
    payload = metadata_envelope(
        run_id=run_id,
        run_name=args.run_name,
        run_dir=args.output_dir / run_id,
        script_path=__file__,
        packages=CHRONOS_PACKAGES,
    )
    payload.update({
        "args": {
            "resolution": args.resolution,
            "context_hours": args.context_hours,
            "requested_context_steps": chronos_extreme.requested_context_steps(args.context_hours, args.resolution),
            "effective_context_steps": chronos_extreme.effective_context_steps(
                args.context_hours,
                args.resolution,
                args.max_context_steps,
            ),
            "context_capped": chronos_extreme.requested_context_steps(args.context_hours, args.resolution)
            > args.max_context_steps,
            "prediction_hours": args.prediction_hours,
            "prediction_steps": chronos_extreme.prediction_steps(args.prediction_hours, args.resolution),
            "forecast_every_hours": args.forecast_every_hours,
            "forecast_every_steps": chronos_extreme.prediction_steps(args.forecast_every_hours, args.resolution),
            "max_forecast_starts": args.max_forecast_starts,
            "model_path": args.model_path,
            "device_map": args.device_map,
            "weather_columns": weather_columns,
            "output_dir": str(args.output_dir),
            "wandb_enabled": not args.disable_wandb,
        },
        "evaluation": {
            "policy": "latest_forecast_wins_within_selected_weeks",
            "raw_predictions_csv": "all issued forecast horizons",
            "operational_predictions_csv": "one latest forecast per selected-week timestamp",
            "metrics_summary_csv": "computed from operational_predictions.csv",
        },
        "forecast_windowing": {
            "mode": "overlapping_within_selected_week",
            "window_hours": args.prediction_hours,
            "step_hours": args.forecast_every_hours,
            "week_duration_hours": 168,
            "evaluation_policy": "latest_forecast_wins",
        },
        "selected_forecast_weeks": selected_weeks.to_dict(orient="records"),
        "n_issued_forecast_starts": len(starts),
        "data": {
            "heat_path": str(args.heat_path),
            "heat_sha256": chronos_extreme.sha256_file(args.heat_path),
            "weather_path": str(args.weather_path),
            "weather_sha256": chronos_extreme.sha256_file(args.weather_path),
            "selected_weeks_path": str(args.selected_weeks_path),
            "selected_weeks_sha256": chronos_extreme.sha256_file(args.selected_weeks_path),
        },
        "model": {
            "name": chronos_extreme.MODEL_NAME,
            "implementation": "chronos.Chronos2Pipeline",
            "model_path": args.model_path,
            "direct_chronos": True,
            "autogluon_used": False,
        },
        "time_handling": {
            "timezone": chronos_extreme.TIMEZONE,
            "model_timestamp": "synthetic continuous naive timestamp axis",
            "output_timestamp": "original Europe/Berlin timestamp retained in outputs",
        },
        "resolution_handling": {
            "resolution": args.resolution,
            "step": str(chronos_extreme.RESOLUTION_TO_STEP[args.resolution]),
            "steps_per_hour": chronos_extreme.STEPS_PER_HOUR_BY_RESOLUTION[args.resolution],
        },
        "quantiles": chronos_extreme.DEFAULT_QUANTILES,
        "interval_definitions": chronos_extreme.INTERVAL_DEFINITIONS,
        "wandb": {
            "run_id": getattr(wandb_run, "id", None) if wandb_run is not None else None,
            "run_name": getattr(wandb_run, "name", None) if wandb_run is not None else None,
        },
        "total_seconds": total_seconds,
    })
    return payload


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()
    weather_columns = parse_weather_columns(args.weather_columns)
    step = chronos_extreme.RESOLUTION_TO_STEP[args.resolution]

    selected_weeks = load_selected_weeks(args.selected_weeks_path)
    starts = forecast_starts_for_weeks(selected_weeks, args.forecast_every_hours)
    if args.max_forecast_starts is not None:
        if args.max_forecast_starts <= 0:
            raise ValueError(f"max-forecast-starts must be positive, got {args.max_forecast_starts}.")
        starts = starts[: args.max_forecast_starts]
    if not starts:
        raise ValueError("No forecast starts generated.")

    print(f"Model: {chronos_extreme.MODEL_NAME}")
    print(f"Model path: {args.model_path}")
    print(f"Forecast starts requested: {len(starts):,}")
    print("Forecast windowing: overlapping issued forecasts; evaluation uses latest forecast wins inside selected weeks")
    print(f"Resolution: {args.resolution}")
    print(f"Prediction hours: {args.prediction_hours}")
    print(f"Forecast every hours: {args.forecast_every_hours}")
    print(f"Context hours: {args.context_hours}")
    print(f"Weather columns: {', '.join(weather_columns)}")

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
        print("Dry run: validating forecast windows without loading Chronos-2.")
        for idx, (selection, week_id, _, _, forecast_start) in enumerate(starts, start=1):
            context_df, future_df, actual, info = chronos_extreme.build_window(
                data,
                forecast_start,
                args,
                weather_columns,
            )
            print(
                f"[{idx}/{len(starts)}] {selection} {week_id} {forecast_start}: "
                f"context_rows={len(context_df)}, future_rows={len(future_df)}, "
                f"actual_rows={len(actual)}, capped={info['context_capped']}"
            )
        print("Dry run complete. No predictions or outputs written.")
        return

    feature_label = "_".join(weather_columns)
    cap_label = ""
    if chronos_extreme.requested_context_steps(args.context_hours, args.resolution) > args.max_context_steps:
        cap_label = f"_cap{args.max_context_steps}steps"
    default_run_name = (
        f"chronos2_extreme_{args.resolution}_pred{args.prediction_hours}h_"
        f"every{args.forecast_every_hours}h_context{args.context_hours}h{cap_label}_{feature_label}"
    )
    run_id = make_run_id(args.run_id, args.run_name or default_run_name)
    run_dir = args.output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    pipeline = chronos_extreme.initialize_pipeline(args)
    wandb_run = initialize_wandb(
        args,
        run_id,
        {
            "run_id": run_id,
            "run_name": args.run_name,
            "model": chronos_extreme.MODEL_NAME,
            "model_path": args.model_path,
            "resolution": args.resolution,
            "context_hours": args.context_hours,
            "requested_context_steps": chronos_extreme.requested_context_steps(
                args.context_hours,
                args.resolution,
            ),
            "max_context_steps": args.max_context_steps,
            "effective_context_steps": chronos_extreme.effective_context_steps(
                args.context_hours,
                args.resolution,
                args.max_context_steps,
            ),
            "prediction_hours": args.prediction_hours,
            "prediction_steps": chronos_extreme.prediction_steps(args.prediction_hours, args.resolution),
            "forecast_every_hours": args.forecast_every_hours,
            "forecast_every_steps": chronos_extreme.prediction_steps(
                args.forecast_every_hours,
                args.resolution,
            ),
            "evaluation_policy": "latest_forecast_wins_within_selected_weeks",
            "weather_columns": weather_columns,
            "selected_weeks_path": str(args.selected_weeks_path),
            "selected_forecast_weeks": selected_weeks.to_dict(orient="records"),
        },
        tags=("chronos2", "direct", "extreme_weeks", "overlapping_horizon"),
    )

    raw_forecasts = []
    for idx, (selection, week_id, week_start, week_end, forecast_start) in enumerate(starts, start=1):
        print(f"[{idx}/{len(starts)}] Predicting {selection} {week_id}: {forecast_start}", flush=True)
        issued, metrics = chronos_extreme.run_one_forecast(
            pipeline=pipeline,
            data=data,
            selection=selection,
            week_id=week_id,
            forecast_start=forecast_start,
            args=args,
            weather_columns=weather_columns,
            data_loading_seconds=data_loading_seconds,
        )
        issued = add_week_bounds(issued, week_start=week_start, week_end=week_end)
        raw_forecasts.append(issued)
        if wandb_run is not None:
            wandb_run.log(chronos_extreme.wandb_forecast_metrics(metrics), step=idx)
        print(
            f"[{idx}/{len(starts)}] issued RMSE={metrics['RMSE']:.3f}, "
            f"issued CVRMSE={metrics['CVRMSE_percent']:.2f}%, "
            f"prediction={metrics['prediction_seconds']:.2f}s",
            flush=True,
        )

    raw = pd.concat(raw_forecasts, ignore_index=True)
    operational = latest_forecast_wins(raw)
    metrics = calculate_per_forecast_start_metrics(operational)
    summary = chronos_extreme.calculate_summary_metrics(operational)
    summary.insert(1, "evaluation_policy", "latest_forecast_wins")

    raw.to_csv(run_dir / "raw_predictions.csv", index=False)
    operational.to_csv(run_dir / "operational_predictions.csv", index=False)
    metrics.to_csv(run_dir / "metrics_per_forecast_start.csv", index=False)
    summary.to_csv(run_dir / "metrics_summary.csv", index=False)

    if wandb_run is not None:
        chronos_extreme.log_wandb_summary(wandb_run, summary)
        wandb_run.finish()

    total_seconds = time.perf_counter() - total_start
    write_metadata(
        run_dir / "run_metadata.json",
        build_metadata(args, run_id, selected_weeks, starts, weather_columns, total_seconds, wandb_run),
    )

    print(f"Issued forecast rows: {len(raw):,}")
    print(f"Operational evaluated rows: {len(operational):,}")
    print(f"Saved raw predictions: {run_dir / 'raw_predictions.csv'}")
    print(f"Saved operational predictions: {run_dir / 'operational_predictions.csv'}")
    print(f"Saved per-start metrics: {run_dir / 'metrics_per_forecast_start.csv'}")
    print(f"Saved summary metrics: {run_dir / 'metrics_summary.csv'}")
    print(f"Saved metadata: {run_dir / 'run_metadata.json'}")
    print(f"Total seconds: {total_seconds:.2f}")


if __name__ == "__main__":
    main()
