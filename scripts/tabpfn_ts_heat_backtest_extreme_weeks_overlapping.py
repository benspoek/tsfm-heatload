from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from full_year_forecasting_utils import (
    DEFAULT_SELECTED_WEEKS_PATH,
    forecast_starts_for_selected_weeks,
    load_selected_weeks,
    local_midnight,
    validate_complete_year,
)
import tabpfn_ts_heat_backtest_extreme_weeks_2024 as extreme
import tabpfn_ts_heat_forecast as forecast
from utils import (
    TABPFN_PACKAGES,
    initialize_wandb,
    make_run_id,
    metadata_envelope,
    parse_weather_columns,
    write_metadata,
)


DEFAULT_HEAT_PATH = forecast.DEFAULT_HEAT_PATH
DEFAULT_WEATHER_PATH = forecast.DEFAULT_WEATHER_PATH
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/extreme_weeks_2024")
DEFAULT_WANDB_PROJECT = "timeseries-forecasting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run overlapping TabPFN-TS forecasts on the selected extreme weeks and "
            "evaluate them as a latest-forecast-wins operational trajectory."
        )
    )
    parser.add_argument("--heat-path", type=Path, default=DEFAULT_HEAT_PATH)
    parser.add_argument("--weather-path", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--selected-weeks-path", type=Path, default=DEFAULT_SELECTED_WEEKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resolution", choices=tuple(forecast.STEPS_PER_HOUR_BY_RESOLUTION), default="hourly")
    parser.add_argument("--context-hours", type=int, default=2016)
    parser.add_argument("--prediction-hours", type=int, default=24)
    parser.add_argument("--forecast-every-hours", type=int, default=4)
    parser.add_argument("--weather-columns", default="temperature")
    parser.add_argument("--mode", choices=("CLIENT", "LOCAL"), default="LOCAL")
    parser.add_argument("--max-forecast-starts", type=int, default=None)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def patch_weather_columns(columns: list[str]) -> None:
    forecast.WEATHER_COLUMNS = columns
    extreme.WEATHER_COLUMNS = columns


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
    week_start_naive = week_start.tz_localize(None)
    week_end_naive = week_end.tz_localize(None)
    out.insert(3, "week_start", week_start_naive)
    out.insert(4, "week_end", week_end_naive)
    out["within_selected_week"] = (out["timestamp"] >= week_start_naive) & (out["timestamp"] < week_end_naive)
    return out


def latest_forecast_wins(raw: pd.DataFrame) -> pd.DataFrame:
    within = raw[raw["within_selected_week"]].copy()
    if within.empty:
        raise ValueError("No issued forecast rows fall inside the selected weeks.")
    within = within.sort_values(["selection", "week_id", "timestamp", "forecast_start"]).reset_index(drop=True)
    operational = within.drop_duplicates(["selection", "week_id", "timestamp"], keep="last").reset_index(drop=True)
    return operational


def calculate_per_forecast_start_metrics(operational: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["selection", "week_id", "forecast_start"]
    for (selection, week_id, forecast_start), group in operational.groupby(group_columns, sort=True):
        row = extreme.calculate_group_metrics("operational_forecast_start", group)
        row.update(
            {
                "selection": selection,
                "week_id": week_id,
                "forecast_start": forecast_start,
                "n_forecast_rows": len(group),
                "n_context_rows": int(group["context_rows"].iloc[0]),
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
        packages=TABPFN_PACKAGES,
    )
    payload.update({
        "args": {
            "resolution": args.resolution,
            "context_hours": args.context_hours,
            "context_days_equivalent": args.context_hours / 24,
            "prediction_hours": args.prediction_hours,
            "prediction_steps": forecast.prediction_steps(args.prediction_hours, args.resolution),
            "forecast_every_hours": args.forecast_every_hours,
            "forecast_every_steps": forecast.prediction_steps(args.forecast_every_hours, args.resolution),
            "max_forecast_starts": args.max_forecast_starts,
            "mode": args.mode,
            "output_dir": str(args.output_dir),
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_enabled": not args.disable_wandb,
        },
        "evaluation": {
            "policy": "latest_forecast_wins_within_selected_weeks",
            "raw_predictions_csv": "all issued forecast horizons",
            "operational_predictions_csv": "one latest forecast per selected-week timestamp",
            "metrics_summary_csv": "computed from operational_predictions.csv",
        },
        "selected_forecast_weeks": selected_weeks.to_dict(orient="records"),
        "n_issued_forecast_starts": len(starts),
        "selected_weeks_source": extreme.selected_weeks_source_metadata(args.selected_weeks_path),
        "data": {
            "heat_path": str(args.heat_path),
            "heat_sha256": extreme.sha256_file(args.heat_path),
            "weather_path": str(args.weather_path),
            "weather_sha256": extreme.sha256_file(args.weather_path),
            "weather_columns": weather_columns,
        },
        "resolution_handling": {
            "resolution": args.resolution,
            "frequency": forecast.RESOLUTION_TO_FREQ[args.resolution],
            "steps_per_hour": forecast.STEPS_PER_HOUR_BY_RESOLUTION[args.resolution],
        },
        "quantiles": extreme.DEFAULT_QUANTILES,
        "interval_definitions": extreme.INTERVAL_DEFINITIONS,
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
    patch_weather_columns(weather_columns)

    selected_weeks = load_selected_weeks(args.selected_weeks_path)
    starts = forecast_starts_for_weeks(selected_weeks, args.forecast_every_hours)
    if args.max_forecast_starts is not None:
        starts = starts[: args.max_forecast_starts]
    if not starts:
        raise ValueError("No forecast starts generated.")

    print(f"Forecast starts requested: {len(starts):,}")
    print(
        "Forecast windowing: overlapping issued forecasts; "
        "evaluation uses latest forecast wins inside selected weeks"
    )
    print(f"Resolution: {args.resolution}")
    print(f"Prediction hours: {args.prediction_hours}")
    print(f"Forecast every hours: {args.forecast_every_hours}")
    print(f"Context hours: {args.context_hours}")
    print(f"Weather columns: {', '.join(weather_columns)}")

    load_start = time.perf_counter()
    data = forecast.load_data(args.heat_path, args.weather_path, resolution=args.resolution)
    data_loading_seconds = time.perf_counter() - load_start
    validate_complete_year(
        data,
        year=2024,
        step=forecast.EXPECTED_STEP_BY_RESOLUTION[args.resolution],
        value_columns=["heat", *weather_columns],
        label="selected-week forecast data",
    )

    context_lengths = []
    for _, _, _, _, forecast_start in starts:
        context_df, future_df, _, _ = forecast.build_windows(
            data,
            forecast_start=forecast_start,
            context_hours=args.context_hours,
            prediction_hours=args.prediction_hours,
            resolution=args.resolution,
        )
        forecast.validate_model_inputs(context_df, future_df)
        context_lengths.append(len(context_df))
    max_context_length = max(context_lengths)
    print(f"TabPFN-TS max_context_length set to: {max_context_length:,}")

    if args.dry_run:
        print("Dry run complete. No predictions or outputs written.")
        return

    default_run_name = (
        f"overlap_{args.resolution}_pred{args.prediction_hours}h_"
        f"every{args.forecast_every_hours}h_context{args.context_hours}h"
    )
    run_id = make_run_id(args.run_id, args.run_name or default_run_name)
    run_dir = args.output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    pipeline = extreme.initialize_pipeline(args.mode, max_context_length=max_context_length)
    wandb_run = initialize_wandb(
        args,
        run_id,
        {
            "run_id": run_id,
            "run_name": args.run_name,
            "resolution": args.resolution,
            "resolution_frequency": forecast.RESOLUTION_TO_FREQ[args.resolution],
            "context_hours": args.context_hours,
            "context_days_equivalent": args.context_hours / 24,
            "prediction_hours": args.prediction_hours,
            "prediction_steps": forecast.prediction_steps(args.prediction_hours, args.resolution),
            "forecast_every_hours": args.forecast_every_hours,
            "forecast_every_steps": forecast.prediction_steps(args.forecast_every_hours, args.resolution),
            "max_forecast_starts": args.max_forecast_starts,
            "evaluation_policy": "latest_forecast_wins_within_selected_weeks",
            "weather_columns": weather_columns,
            "selected_weeks_path": str(args.selected_weeks_path),
            "selected_forecast_weeks": selected_weeks.to_dict(orient="records"),
        },
    )

    raw_forecasts = []
    for idx, (selection, week_id, week_start, week_end, forecast_start) in enumerate(starts, start=1):
        print(f"[{idx}/{len(starts)}] Predicting {selection} {week_id}: {forecast_start}")
        issued, metrics = extreme.predict_one_day(
            pipeline=pipeline,
            data=data,
            selection=selection,
            week_id=week_id,
            forecast_start=forecast_start,
            args=args,
            data_loading_seconds=data_loading_seconds,
        )
        issued = add_week_bounds(issued, week_start=week_start, week_end=week_end)
        raw_forecasts.append(issued)
        if wandb_run is not None:
            wandb_run.log(extreme.wandb_daily_metrics(metrics), step=idx)
        print(
            f"[{idx}/{len(starts)}] issued RMSE={metrics['RMSE']:.3f}, "
            f"issued CVRMSE={metrics['CVRMSE_percent']:.2f}%, "
            f"prediction={metrics['prediction_seconds']:.2f}s"
        )

    raw = pd.concat(raw_forecasts, ignore_index=True)
    operational = latest_forecast_wins(raw)
    metrics = calculate_per_forecast_start_metrics(operational)
    summary = extreme.calculate_summary_metrics(operational)
    summary.insert(1, "evaluation_policy", "latest_forecast_wins")

    raw.to_csv(run_dir / "raw_predictions.csv", index=False)
    operational.to_csv(run_dir / "operational_predictions.csv", index=False)
    metrics.to_csv(run_dir / "metrics_per_forecast_start.csv", index=False)
    summary.to_csv(run_dir / "metrics_summary.csv", index=False)

    if wandb_run is not None:
        extreme.log_wandb_summary(wandb_run, summary)
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
