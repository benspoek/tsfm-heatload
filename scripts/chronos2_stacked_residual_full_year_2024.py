from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import stacked_residual_full_year_2024 as stacked
from full_year_forecasting_utils import (
    HOURLY_STEP,
    QUARTER_STEP,
    STEPS_PER_HOUR,
    UnsupportedDataCadenceError,
    add_synthetic_model_clock,
    load_residual_multiresolution_data,
    sha256_file,
    to_naive_datetime,
    validate_regular_rows,
)
from utils import CHRONOS_PACKAGES, make_run_id, metadata_envelope, write_metadata


DEFAULT_OUTPUT_DIR = Path("outputs/experiments/full_year_2024/stacked_chronos2")
DEFAULT_RUN_NAME = "chronos2_stacked_full_year_2024_base12w_pred24h_every12h_residual7d_pred2h_every1h_temperature"
DEFAULT_MODEL_PATH = "amazon/chronos-2"
DEFAULT_MAX_CONTEXT_STEPS = 8192
MODEL_NAME = "Chronos-2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full-year deterministic stacked direct Chronos-2 forecast for a Munich district heating network. "
            "The base model is hourly with 12-week context and 24h horizon every 12h; "
            "the residual model is 15-minute with 7-day context and 2h horizon every hour."
        )
    )
    parser.add_argument("--heat-path", type=Path, default=stacked.DEFAULT_HEAT_PATH)
    parser.add_argument("--weather-path", type=Path, default=stacked.DEFAULT_WEATHER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--dataset-name", default="munich")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--max-context-steps", type=int, default=DEFAULT_MAX_CONTEXT_STEPS)
    parser.add_argument("--max-forecast-starts", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, start construction, and first windows without running Chronos-2 predictions.",
    )
    return parser.parse_args()


def initialize_pipeline(args: argparse.Namespace):
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install Chronos-2 direct inference support with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return Chronos2Pipeline.from_pretrained(args.model_path, device_map=args.device_map)


def find_column_by_numeric_value(columns: pd.Index, value: float) -> object | None:
    for column in columns:
        try:
            if math.isclose(float(column), value, rel_tol=0.0, abs_tol=1e-12):
                return column
        except (TypeError, ValueError):
            continue
    return None


def flatten_point_prediction(pred_df: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    out = pred_df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    elif "timestamp" not in out.columns:
        out = out.reset_index()
    if "timestamp" not in out.columns:
        raise ValueError(f"Unexpected Chronos-2 prediction output columns: {out.columns.tolist()}")

    if "target" in out.columns:
        value_column = "target"
    elif "mean" in out.columns:
        value_column = "mean"
    else:
        value_column = find_column_by_numeric_value(out.columns, 0.5)
        if value_column is None:
            raise ValueError(
                "Chronos-2 prediction output has no point forecast column. Expected one of "
                f"'target', 'mean', or quantile 0.5. Columns: {out.columns.tolist()}"
            )

    result = out[["timestamp"]].rename(columns={"timestamp": "model_timestamp"}).copy()
    result["model_timestamp"] = to_naive_datetime(result["model_timestamp"])
    result[prediction_column] = pd.to_numeric(out[value_column], errors="raise").to_numpy(dtype=float)
    return result


def predict_point(
    pipeline,
    context_df: pd.DataFrame,
    future_df: pd.DataFrame,
    prediction_column: str,
    max_context_steps: int,
) -> tuple[pd.DataFrame, float]:
    if len(context_df) > max_context_steps:
        raise ValueError(
            "Chronos-2 context would be capped or truncated, which is not allowed for this matched stacked run: "
            f"max_context_steps={max_context_steps}, context rows={len(context_df)}."
        )
    if "target" in future_df.columns:
        raise ValueError("future_df must not contain target.")

    started = time.perf_counter()
    prediction = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=len(future_df),
        quantile_levels=[0.5],
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
    )
    prediction_seconds = time.perf_counter() - started
    return flatten_point_prediction(prediction, prediction_column), prediction_seconds


def run_base_forecasts(
    pipeline,
    hourly: pd.DataFrame,
    starts: pd.DataFrame,
    max_context_steps: int,
) -> pd.DataFrame:
    rows = []
    for idx, start_row in starts.iterrows():
        forecast_start = start_row["forecast_start"]
        print(f"[base {idx + 1}/{len(starts)}] {forecast_start}", flush=True)
        context_df, future_df, future_raw = stacked.build_window(
            data=hourly,
            forecast_start=forecast_start,
            context_hours=stacked.BASE_CONTEXT_HOURS,
            prediction_hours=stacked.BASE_PREDICTION_HOURS,
            step=HOURLY_STEP,
            target_column="heat",
            covariate_columns=stacked.WEATHER_COLUMNS,
        )
        prediction, prediction_seconds = predict_point(
            pipeline,
            context_df=context_df,
            future_df=future_df,
            prediction_column=stacked.BASE_HOURLY_PREDICTION_COLUMN,
            max_context_steps=max_context_steps,
        )
        out = future_raw.merge(prediction, on="model_timestamp", how="left", validate="one_to_one")
        if out[stacked.BASE_HOURLY_PREDICTION_COLUMN].isna().any():
            raise ValueError(f"Missing base Chronos-2 predictions for {forecast_start}.")
        out.insert(0, "model", "hourly_base")
        out.insert(1, "forecast_start", forecast_start)
        out.insert(2, "horizon_step", np.arange(1, len(out) + 1))
        out.insert(3, "horizon_minutes", out["horizon_step"] * 60)
        out.insert(4, "context_rows", len(context_df))
        out.insert(5, "prediction_seconds", prediction_seconds)
        rows.append(out.rename(columns={"heat": "actual_heat_hourly"}))
    return pd.concat(rows, ignore_index=True)


def run_residual_forecasts(
    pipeline,
    quarter: pd.DataFrame,
    raw_base: pd.DataFrame,
    starts: pd.DataFrame,
    max_context_steps: int,
) -> pd.DataFrame:
    rows = []
    for idx, start_row in starts.iterrows():
        forecast_start = start_row["forecast_start"]
        print(f"[residual {idx + 1}/{len(starts)}] {forecast_start}", flush=True)
        context_df, future_df, future_raw = stacked.build_residual_window(quarter, raw_base, forecast_start)
        prediction, prediction_seconds = predict_point(
            pipeline,
            context_df=context_df,
            future_df=future_df,
            prediction_column=stacked.RESIDUAL_PREDICTION_COLUMN,
            max_context_steps=max_context_steps,
        )
        out = future_raw.merge(prediction, on="model_timestamp", how="left", validate="one_to_one")
        if out[stacked.RESIDUAL_PREDICTION_COLUMN].isna().any():
            raise ValueError(f"Missing residual Chronos-2 predictions for {forecast_start}.")
        out.insert(0, "model", "stacked_residual")
        out.insert(1, "forecast_start", forecast_start)
        out.insert(2, "horizon_step", np.arange(1, len(out) + 1))
        out.insert(3, "horizon_minutes", out["horizon_step"] * 15)
        out.insert(4, "context_rows", len(context_df))
        out.insert(5, "prediction_seconds", prediction_seconds)
        out[stacked.STACKED_PREDICTION_COLUMN] = (
            out[stacked.BASE_QUARTER_PREDICTION_COLUMN] + out[stacked.RESIDUAL_PREDICTION_COLUMN]
        )
        rows.append(out.rename(columns={"heat": "actual_heat"}))
    return pd.concat(rows, ignore_index=True)


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
        packages=CHRONOS_PACKAGES,
    )
    doc.update({
        "inputs": {
            "dataset_name": args.dataset_name,
            "heat_path": str(args.heat_path),
            "heat_sha256": sha256_file(args.heat_path),
            "weather_path": str(args.weather_path),
            "weather_sha256": sha256_file(args.weather_path),
            "weather_columns": stacked.WEATHER_COLUMNS,
        },
        "model": {
            "name": MODEL_NAME,
            "implementation": "chronos.Chronos2Pipeline",
            "model_path": args.model_path,
            "device_map": args.device_map,
            "direct_chronos": True,
            "autogluon_used": False,
            "deterministic_point_forecast": "q50/median",
            "quantiles_propagated_to_stacked_predictor": False,
        },
        "evaluation": {
            "year": args.year,
            "resolution": "15min",
            "includes_signal_outage_periods": True,
            "future_covariates_exclude_target": True,
            "timestamp_axis": (
                "Base and residual Chronos-2 calls receive synthetic continuous model clocks. "
                "Real Europe/Berlin timestamps are retained for scheduling, interpolation, and outputs."
            ),
        },
        "stacked_predictor": {
            "base": {
                "resolution": "hourly",
                "context_hours": stacked.BASE_CONTEXT_HOURS,
                "context_rows": stacked.BASE_CONTEXT_HOURS,
                "prediction_hours": stacked.BASE_PREDICTION_HOURS,
                "forecast_every_hours": stacked.BASE_FORECAST_EVERY_HOURS,
                "covariates": stacked.WEATHER_COLUMNS,
                "n_forecast_starts": n_base_starts,
                "context_capped": stacked.BASE_CONTEXT_HOURS > args.max_context_steps,
            },
            "residual": {
                "resolution": "15min",
                "context_hours": stacked.RESIDUAL_CONTEXT_HOURS,
                "context_rows": stacked.RESIDUAL_CONTEXT_HOURS * STEPS_PER_HOUR,
                "prediction_hours": stacked.RESIDUAL_PREDICTION_HOURS,
                "forecast_every_hours": stacked.RESIDUAL_FORECAST_EVERY_HOURS,
                "covariates": stacked.RESIDUAL_COVARIATES,
                "target": "actual heat - available base prediction",
                "n_forecast_starts": n_residual_starts,
                "context_capped": stacked.RESIDUAL_CONTEXT_HOURS * STEPS_PER_HOUR > args.max_context_steps,
            },
            "final_prediction": "base_prediction_15min + predicted_residual",
            "overlap_policy": "latest residual forecast wins",
        },
        "max_context_steps": args.max_context_steps,
        "total_seconds": total_seconds,
    })
    return doc


def main() -> None:
    args = parse_args()
    eval_start, eval_end = stacked.year_bounds(args.year)
    try:
        quarter, hourly = load_residual_multiresolution_data(
            args.heat_path,
            args.weather_path,
            stacked.WEATHER_COLUMNS,
        )
    except UnsupportedDataCadenceError as exc:
        raise SystemExit(str(exc)) from None

    validation_start = eval_start - pd.Timedelta(
        hours=stacked.BASE_CONTEXT_HOURS + stacked.RESIDUAL_CONTEXT_HOURS + stacked.BASE_PREDICTION_HOURS
    )
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

    base_start_min = eval_start - pd.Timedelta(
        hours=stacked.RESIDUAL_CONTEXT_HOURS + stacked.BASE_PREDICTION_HOURS
    )
    base_starts = stacked.hourly_start_rows(
        hourly,
        start=base_start_min,
        end_exclusive=eval_end,
        every_hours=stacked.BASE_FORECAST_EVERY_HOURS,
        horizon_hours=stacked.BASE_PREDICTION_HOURS,
        max_count=None if args.max_forecast_starts is None else args.max_forecast_starts + 20,
    )
    residual_starts = stacked.residual_start_rows(
        quarter,
        start=eval_start,
        end_exclusive=eval_end,
        every_hours=stacked.RESIDUAL_FORECAST_EVERY_HOURS,
        horizon_hours=stacked.RESIDUAL_PREDICTION_HOURS,
        max_count=args.max_forecast_starts,
    )
    if base_starts.empty or residual_starts.empty:
        raise ValueError("No base or residual forecast starts were constructed.")

    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {MODEL_NAME}")
    print(f"Model path: {args.model_path}")
    print(f"Base starts: {len(base_starts):,}")
    print(f"Residual starts: {len(residual_starts):,}")
    print(f"Evaluation period: {eval_start} <= timestamp < {eval_end}")
    print(f"Base context rows: {stacked.BASE_CONTEXT_HOURS:,}")
    print(f"Residual context rows: {stacked.RESIDUAL_CONTEXT_HOURS * STEPS_PER_HOUR:,}")
    print(f"Max context steps: {args.max_context_steps:,}")

    if args.dry_run:
        n_base, n_residual = stacked.validate_forecast_windows(quarter, hourly, base_starts, residual_starts)
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

    pipeline = initialize_pipeline(args)
    raw_base = run_base_forecasts(pipeline, hourly, base_starts, args.max_context_steps)
    raw_residual = run_residual_forecasts(pipeline, quarter, raw_base, residual_starts, args.max_context_steps)
    eval_cutoff = residual_starts["forecast_start"].max() + pd.Timedelta(hours=stacked.RESIDUAL_PREDICTION_HOURS)
    effective_eval_end = min(eval_end, eval_cutoff)
    raw_predictions = stacked.build_final_predictions(
        quarter,
        raw_residual,
        eval_start=eval_start,
        eval_end=effective_eval_end,
    )

    required = [
        stacked.BASE_QUARTER_PREDICTION_COLUMN,
        stacked.RESIDUAL_PREDICTION_COLUMN,
        stacked.STACKED_PREDICTION_COLUMN,
    ]
    if raw_predictions[required].isna().any().any():
        missing_counts = raw_predictions[required].isna().sum().to_dict()
        raise ValueError(f"Missing final prediction columns: {missing_counts}")

    summary = stacked.calculate_summary(raw_predictions, raw_base, raw_residual)
    per_start = stacked.calculate_per_start_metrics(raw_residual)

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
