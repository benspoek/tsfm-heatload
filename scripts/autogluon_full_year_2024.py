from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from autogluon_forecasting_utils import (
    align_prediction_with_actual,
    build_autogluon_prediction_inputs,
    prediction_to_frame,
    to_autogluon_frame,
)
from full_year_forecasting_utils import (
    HOURLY_STEP,
    TIMEZONE,
    add_synthetic_model_clock,
    calculate_summary,
    forecast_starts_from_rows,
    load_merged_data,
    metric_values,
    sha256_file,
    validate_regular_rows,
    validate_train_test_boundary,
)
from utils import (
    AUTOGLUON_PACKAGES,
    make_run_id,
    metadata_envelope,
    parse_weather_columns,
    write_metadata,
)


DEFAULT_HEAT_PATH = Path("flensburg/demand/heat/heat_dh.csv")
DEFAULT_WEATHER_PATH = Path("flensburg/weather/flensburg_weather_temperature.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/full_year_2024/autogluon")
DEFAULT_RUN_NAME = "autogluon_full_year_2024_hourly_pred24h_temperature"
DEFAULT_TIME_LIMIT_SECONDS = 13_800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train AutoGluon TimeSeries on 2023 heat-load targets and evaluate "
            "full-year 2024 rolling 24h forecasts. Default uses ambient temperature only."
        )
    )
    parser.add_argument("--heat-path", type=Path, default=DEFAULT_HEAT_PATH)
    parser.add_argument("--weather-path", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--dataset-name", default="flensburg")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--train-year", type=int, default=2023)
    parser.add_argument("--weather-columns", default="temperature")
    parser.add_argument("--prediction-hours", type=int, default=24)
    parser.add_argument("--forecast-every-hours", type=int, default=24)
    parser.add_argument("--time-limit-seconds", type=int, default=DEFAULT_TIME_LIMIT_SECONDS)
    parser.add_argument("--num-val-windows", type=int, default=1)
    parser.add_argument("--disable-ensemble", action="store_true")
    parser.add_argument(
        "--disable-neural",
        action="store_true",
        help="Disable neural AutoGluon TimeSeries models and keep only tabular/statistical baselines.",
    )
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument("--max-forecast-starts", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, train/test split, and forecast starts without fitting AutoGluon.",
    )
    return parser.parse_args()


def build_autogluon_hyperparameters(
    include_neural: bool = True,
) -> dict[str, object]:
    hyperparameters: dict[str, object] = {
        "DirectTabular": [
            {
                "model_name": "GBM",
                "model_hyperparameters": {},
                "ag_args": {"name_suffix": "LightGBM"},
            },
            {
                "model_name": "XGB",
                "model_hyperparameters": {},
                "ag_args": {"name_suffix": "XGBoost"},
            },
        ],
        "RecursiveTabular": [
            {
                "model_name": "GBM",
                "model_hyperparameters": {},
                "ag_args": {"name_suffix": "LightGBM"},
            },
        ],
        "ETS": {},
        "AutoETS": {},
        "AutoARIMA": {},
        "SeasonalNaive": {},
        "Naive": {},
    }
    if include_neural:
        hyperparameters.update(
            {
                "DeepAR": {},
                "TemporalFusionTransformer": {},
                "SimpleFeedForward": {},
                "DLinear": {},
            }
        )
    return hyperparameters


def build_autogluon_hyperparameters_from_args(args: argparse.Namespace) -> dict[str, object]:
    return build_autogluon_hyperparameters(
        include_neural=not args.disable_neural,
    )


def to_ts_dataframe(df: pd.DataFrame):
    try:
        from autogluon.timeseries import TimeSeriesDataFrame
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install it with: python -m pip install -r requirements.txt"
        ) from exc

    return TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")


def initialize_predictor(run_dir: Path, prediction_hours: int, weather_columns: list[str], args: argparse.Namespace):
    try:
        from autogluon.timeseries import TimeSeriesPredictor
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install it with: python -m pip install -r requirements.txt"
        ) from exc

    return TimeSeriesPredictor(
        path=str(run_dir / "autogluon_models"),
        prediction_length=prediction_hours,
        freq="h",
        target="target",
        known_covariates_names=weather_columns,
        eval_metric="RMSE",
        verbosity=args.verbosity,
    )


def build_forecast_output(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    forecast_start: pd.Timestamp,
    prediction_seconds: float,
) -> pd.DataFrame:
    out = align_prediction_with_actual(predicted, actual, forecast_start)
    out = out.drop(columns=["model_timestamp"])
    out.insert(0, "forecast_start", forecast_start.isoformat())
    out.insert(1, "model", out.pop("model"))
    out.insert(2, "horizon_step", np.arange(1, len(out) + 1))
    out.insert(3, "horizon_minutes", out["horizon_step"] * 60)
    out["resolution"] = "hourly"
    out["prediction_hours"] = len(out)
    out["prediction_steps"] = len(out)
    out["error"] = out["predicted_heat"] - out["actual_heat"]
    out["absolute_error"] = out["error"].abs()
    denominator = out["actual_heat"].abs() + out["predicted_heat"].abs()
    out["sape"] = np.where(denominator > 0, 2 * out["absolute_error"] / denominator, np.nan)
    out["prediction_seconds"] = prediction_seconds
    return out


def calculate_metrics(forecast: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {
        "model": str(forecast["model"].iloc[0]),
        "forecast_start": str(forecast["forecast_start"].iloc[0]),
        "resolution": "hourly",
        "prediction_hours": int(forecast["prediction_hours"].iloc[0]),
        "prediction_steps": int(forecast["prediction_steps"].iloc[0]),
        "n_forecast_rows": int(len(forecast)),
        "prediction_seconds": float(forecast["prediction_seconds"].iloc[0]),
    }
    row.update(metric_values(forecast["actual_heat"], forecast["predicted_heat"]))
    return row


def write_partial_outputs(run_dir: Path, raw_forecasts: list[pd.DataFrame], metric_rows: list[dict[str, object]]) -> None:
    if raw_forecasts:
        pd.concat(raw_forecasts, ignore_index=True).to_csv(run_dir / "raw_predictions.partial.csv", index=False)
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(run_dir / "metrics_per_forecast_start.partial.csv", index=False)


def build_metadata(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    weather_columns: list[str],
    n_forecast_starts: int,
    fit_seconds: float,
    models: list[str],
    total_seconds: float,
) -> dict[str, object]:
    doc = metadata_envelope(
        run_id=run_id,
        run_name=args.run_name,
        run_dir=run_dir,
        script_path=__file__,
        packages=AUTOGLUON_PACKAGES,
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
        "split": {
            "fit_target_year": args.train_year,
            "evaluation_year": args.year,
            "fit_target_period": f"{args.train_year}-01-01 <= timestamp < {args.train_year + 1}-01-01",
            "prediction_policy": (
                "Predictor is fit on training-year target values only. During evaluation-year prediction, "
                "observed target history before each forecast start is provided as operational context."
            ),
        },
        "autogluon": {
            "prediction_length": args.prediction_hours,
            "freq": "h",
            "timestamp_axis": (
                "AutoGluon is fit and predicted on a synthetic continuous hourly model clock. "
                "Real Europe/Berlin timestamps are retained in outputs. This avoids missing "
                "civil timestamps at daylight-saving-time transitions."
            ),
            "eval_metric": "RMSE",
            "time_limit_seconds": args.time_limit_seconds,
            "num_val_windows": args.num_val_windows,
            "enable_ensemble": not args.disable_ensemble,
            "include_neural_models": not args.disable_neural,
            "hyperparameters": build_autogluon_hyperparameters_from_args(args),
            "models": models,
            "fit_seconds": fit_seconds,
        },
        "forecasting": {
            "forecast_every_hours": args.forecast_every_hours,
            "forecast_cadence": "row-based over the evaluation year",
            "n_forecast_starts": n_forecast_starts,
            "future_covariates_exclude_target": True,
        },
        "total_seconds": total_seconds,
    })
    return doc


def main() -> None:
    args = parse_args()
    weather_columns = parse_weather_columns(args.weather_columns)
    data = load_merged_data(args.heat_path, args.weather_path, weather_columns, step=HOURLY_STEP)
    validate_regular_rows(data, HOURLY_STEP, "hourly merged data")
    validate_train_test_boundary(
        data,
        train_year=args.train_year,
        test_year=args.year,
        step=HOURLY_STEP,
        value_columns=["heat", *weather_columns],
    )
    data = add_synthetic_model_clock(data, HOURLY_STEP)

    train_data = data[
        (data["timestamp"] >= pd.Timestamp(f"{args.train_year}-01-01", tz=TIMEZONE))
        & (data["timestamp"] < pd.Timestamp(f"{args.train_year + 1}-01-01", tz=TIMEZONE))
    ]
    if train_data.empty:
        raise ValueError(f"Training data for {args.train_year} is empty.")
    if train_data["timestamp"].dt.year.ne(args.train_year).any():
        raise ValueError(f"Training data contains timestamps outside {args.train_year}.")

    starts = forecast_starts_from_rows(
        data=data,
        year=args.year,
        horizon_steps=args.prediction_hours,
        stride_steps=args.forecast_every_hours,
    )
    if args.max_forecast_starts is not None:
        starts = starts.head(args.max_forecast_starts)
    if starts.empty:
        raise ValueError(f"No forecast starts found for {args.year}.")

    print(f"Dataset: {args.dataset_name}", flush=True)
    print(f"Weather columns: {', '.join(weather_columns)}", flush=True)
    print(f"Training rows: {len(train_data):,}", flush=True)
    print(f"Forecast starts: {len(starts):,}", flush=True)
    print(f"First start: {starts['forecast_start'].iloc[0]}", flush=True)
    print(f"Last start: {starts['forecast_start'].iloc[-1]}", flush=True)

    if args.dry_run:
        for forecast_start in starts["forecast_start"]:
            build_autogluon_prediction_inputs(
                data=data,
                forecast_start=forecast_start,
                prediction_steps=args.prediction_hours,
                history_start=pd.Timestamp(f"{args.train_year}-01-01", tz=TIMEZONE),
                evaluation_year=args.year,
                weather_columns=weather_columns,
                step=HOURLY_STEP,
            )
        print("Dry run complete. No AutoGluon fit or outputs written.", flush=True)
        return

    total_start = time.perf_counter()
    run_id = make_run_id(args.run_id, args.run_name)
    run_dir = args.output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    train_ts = to_ts_dataframe(to_autogluon_frame(train_data, weather_columns, include_target=True))
    predictor = initialize_predictor(run_dir, args.prediction_hours, weather_columns, args)
    hyperparameters = build_autogluon_hyperparameters_from_args(args)

    fit_start = time.perf_counter()
    predictor.fit(
        train_data=train_ts,
        time_limit=args.time_limit_seconds,
        hyperparameters=hyperparameters,
        enable_ensemble=not args.disable_ensemble,
        num_val_windows=args.num_val_windows,
    )
    fit_seconds = time.perf_counter() - fit_start

    predictor.leaderboard(display=False).to_csv(
        run_dir / "autogluon_validation_leaderboard.csv",
        index=False,
    )

    models = list(predictor.model_names())
    print(f"Models available for prediction: {', '.join(models)}", flush=True)
    if not any("GBM" in model or "LightGBM" in model for model in models):
        raise RuntimeError(f"Expected a LightGBM/GBM model in AutoGluon models, got: {models}")
    if not any("XGB" in model or "XGBoost" in model for model in models):
        raise RuntimeError(f"Expected an XGBoost/XGB model in AutoGluon models, got: {models}")

    raw_forecasts = []
    metric_rows = []
    for forecast_idx, (_, row) in enumerate(starts.iterrows(), start=1):
        forecast_start = row["forecast_start"]
        print(f"[{forecast_idx}/{len(starts)}] Forecast start {forecast_start}", flush=True)
        history_frame, future_covariates, actual = build_autogluon_prediction_inputs(
            data=data,
            forecast_start=forecast_start,
            prediction_steps=args.prediction_hours,
            history_start=pd.Timestamp(f"{args.train_year}-01-01", tz=TIMEZONE),
            evaluation_year=args.year,
            weather_columns=weather_columns,
            step=HOURLY_STEP,
        )
        history_ts = to_ts_dataframe(history_frame)
        future_covariates_ts = to_ts_dataframe(future_covariates)
        for model_idx, model in enumerate(models, start=1):
            print(
                f"  [{forecast_idx}/{len(starts)} model {model_idx}/{len(models)}] Starting {model}",
                flush=True,
            )
            prediction_start = time.perf_counter()
            prediction = predictor.predict(history_ts, known_covariates=future_covariates_ts, model=model)
            prediction_seconds = time.perf_counter() - prediction_start
            forecast = build_forecast_output(
                predicted=prediction_to_frame(prediction, model_name=model),
                actual=actual,
                forecast_start=forecast_start,
                prediction_seconds=prediction_seconds,
            )
            raw_forecasts.append(forecast)
            metrics = calculate_metrics(forecast)
            metric_rows.append(metrics)
            print(
                f"  {model}: RMSE={metrics['RMSE']:.3f}, "
                f"CVRMSE={metrics['CVRMSE_percent']:.2f}%, prediction={prediction_seconds:.2f}s",
                flush=True,
            )
        write_partial_outputs(run_dir, raw_forecasts, metric_rows)

    raw = pd.concat(raw_forecasts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = calculate_summary(raw)
    leaderboard = (
        summary[summary["metric_scope"] == "all"]
        .sort_values(["RMSE", "CVRMSE_percent"], ascending=True)
        .reset_index(drop=True)
    )
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))

    raw.to_csv(run_dir / "raw_predictions.csv", index=False)
    metrics.to_csv(run_dir / "metrics_per_forecast_start.csv", index=False)
    summary.to_csv(run_dir / "metrics_summary.csv", index=False)
    leaderboard.to_csv(run_dir / "autogluon_leaderboard.csv", index=False)
    for partial_path in [
        run_dir / "raw_predictions.partial.csv",
        run_dir / "metrics_per_forecast_start.partial.csv",
    ]:
        if partial_path.exists():
            partial_path.unlink()

    total_seconds = time.perf_counter() - total_start
    write_metadata(
        run_dir / "run_metadata.json",
        build_metadata(
            args=args,
            run_id=run_id,
            run_dir=run_dir,
            weather_columns=weather_columns,
            n_forecast_starts=len(starts),
            fit_seconds=fit_seconds,
            models=models,
            total_seconds=total_seconds,
        ),
    )

    print(f"Saved run directory: {run_dir}")
    print(f"Saved raw predictions: {run_dir / 'raw_predictions.csv'}")
    print(f"Saved per-start metrics: {run_dir / 'metrics_per_forecast_start.csv'}")
    print(f"Saved summary metrics: {run_dir / 'metrics_summary.csv'}")
    print(f"Saved leaderboard: {run_dir / 'autogluon_leaderboard.csv'}")
    print(f"Saved metadata: {run_dir / 'run_metadata.json'}")
    print(f"Total seconds: {total_seconds:.2f}")


if __name__ == "__main__":
    main()
