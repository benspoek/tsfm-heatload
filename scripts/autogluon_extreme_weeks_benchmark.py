from __future__ import annotations

import argparse
import math
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
    DEFAULT_SELECTED_WEEKS_PATH,
    HOURLY_STEP,
    TIMEZONE,
    add_synthetic_model_clock,
    data_slice_by_year,
    forecast_starts_for_selected_weeks,
    load_merged_data,
    load_selected_weeks,
    sha256_file,
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
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/autogluon_extreme_weeks_2024")
DEFAULT_PREDICTION_HOURS = 24
DEFAULT_TIME_LIMIT_SECONDS = 13_800
DEFAULT_WEATHER_COLUMNS = ("temperature",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train AutoGluon TimeSeries baselines on 2023 heat-load data and "
            "evaluate non-overlapping 24h forecasts on selected 2024 extreme weeks."
        )
    )
    parser.add_argument("--heat-path", type=Path, default=DEFAULT_HEAT_PATH)
    parser.add_argument("--weather-path", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--selected-weeks-path", type=Path, default=DEFAULT_SELECTED_WEEKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default="autogluon_extreme_weeks_hourly_pred24h")
    parser.add_argument("--prediction-hours", type=int, default=DEFAULT_PREDICTION_HOURS)
    parser.add_argument(
        "--weather-columns",
        default=",".join(DEFAULT_WEATHER_COLUMNS),
        help="Comma-separated known weather covariates to use. Default uses temperature.",
    )
    parser.add_argument("--time-limit-seconds", type=int, default=DEFAULT_TIME_LIMIT_SECONDS)
    parser.add_argument("--max-forecast-starts", type=int, default=None)
    parser.add_argument("--num-val-windows", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs, train/test split, forecast windows, and covariates without training AutoGluon.",
    )
    parser.add_argument("--disable-ensemble", action="store_true")
    parser.add_argument(
        "--disable-neural",
        action="store_true",
        help="Disable neural AutoGluon TimeSeries models and keep only tabular/statistical baselines.",
    )
    parser.add_argument("--verbosity", type=int, default=2)
    return parser.parse_args()


def to_ts_dataframe(df: pd.DataFrame):
    try:
        from autogluon.timeseries import TimeSeriesDataFrame
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install AutoGluon TimeSeries, for example: "
            "python -m pip install -r requirements.txt"
        ) from exc

    return TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp",
    )


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


def initialize_predictor(
    run_dir: Path,
    prediction_hours: int,
    weather_columns: list[str],
    args: argparse.Namespace,
):
    try:
        from autogluon.timeseries import TimeSeriesPredictor
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install AutoGluon TimeSeries, for example: "
            "python -m pip install -r requirements.txt"
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


def validate_dry_run(
    data: pd.DataFrame,
    train_data: pd.DataFrame,
    starts: list[tuple[str, str, pd.Timestamp]],
    prediction_hours: int,
    weather_columns: list[str],
) -> None:
    train_frame = to_autogluon_frame(train_data, weather_columns, include_target=True)
    if train_data["timestamp"].dt.year.ne(2023).any():
        raise ValueError("Dry-run validation failed: real training timestamps are not limited to 2023.")
    model_deltas = train_frame["timestamp"].diff().dropna()
    if not model_deltas.eq(HOURLY_STEP).all():
        raise ValueError("Dry-run validation failed: AutoGluon training clock is not regular hourly data.")

    for start_number, (selection, week_id, forecast_start) in enumerate(starts, start=1):
        history_frame, future_covariates, actual = build_autogluon_prediction_inputs(
            data=data,
            forecast_start=forecast_start,
            prediction_steps=prediction_hours,
            history_start=pd.Timestamp("2023-01-01", tz=TIMEZONE),
            evaluation_year=2024,
            weather_columns=weather_columns,
            step=HOURLY_STEP,
        )
        if "target" in future_covariates.columns:
            raise ValueError(f"Dry-run validation failed: future covariates contain target at {forecast_start}.")
        if len(future_covariates) != prediction_hours:
            raise ValueError(
                f"Dry-run validation failed: future covariates for {forecast_start} have "
                f"{len(future_covariates)} rows, expected {prediction_hours}."
            )
        if len(actual) != prediction_hours:
            raise ValueError(
                f"Dry-run validation failed: actual horizon for {forecast_start} has "
                f"{len(actual)} rows, expected {prediction_hours}."
            )
        print(
            f"[dry-run {start_number}/{len(starts)}] {selection} {week_id} {forecast_start}: "
            f"history_rows={len(history_frame):,}, future_rows={len(future_covariates):,}"
        )


def build_forecast_output(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    selection: str,
    week_id: str,
    forecast_start: pd.Timestamp,
    prediction_seconds: float,
) -> pd.DataFrame:
    out = align_prediction_with_actual(predicted, actual, forecast_start)
    out = out.drop(columns=["model_timestamp"])
    out.insert(0, "selection", selection)
    out.insert(1, "week_id", week_id)
    out.insert(2, "forecast_start", forecast_start.isoformat())
    out.insert(3, "model", out.pop("model"))
    out.insert(4, "horizon_step", np.arange(1, len(out) + 1))
    out.insert(5, "horizon_minutes", out["horizon_step"] * 60)
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
    error = forecast["error"].to_numpy(dtype=float)
    actual = forecast["actual_heat"].to_numpy(dtype=float)
    absolute_error = np.abs(error)
    rmse = math.sqrt(float(np.mean(error**2)))
    actual_mean = float(np.mean(actual))
    total_sum_of_squares = float(np.sum((actual - actual_mean) ** 2))
    residual_sum_of_squares = float(np.sum(error**2))
    return {
        "model": str(forecast["model"].iloc[0]),
        "selection": str(forecast["selection"].iloc[0]),
        "week_id": str(forecast["week_id"].iloc[0]),
        "forecast_start": str(forecast["forecast_start"].iloc[0]),
        "resolution": "hourly",
        "prediction_hours": int(forecast["prediction_hours"].iloc[0]),
        "prediction_steps": int(forecast["prediction_steps"].iloc[0]),
        "n_forecast_rows": int(len(forecast)),
        "MAE": float(np.mean(absolute_error)),
        "RMSE": rmse,
        "R2": float(1 - residual_sum_of_squares / total_sum_of_squares) if total_sum_of_squares > 0 else np.nan,
        "CVRMSE_percent": float(rmse / actual_mean * 100) if actual_mean != 0 else np.nan,
        "sMAPE_percent": float(np.nanmean(forecast["sape"]) * 100),
        "mean_error": float(np.mean(error)),
        "median_absolute_error": float(np.median(absolute_error)),
        "max_absolute_error": float(np.max(absolute_error)),
        "prediction_seconds": float(forecast["prediction_seconds"].iloc[0]),
    }


def calculate_group_metrics(scope: str, group: pd.DataFrame) -> dict[str, object]:
    error = group["error"].to_numpy(dtype=float)
    actual = group["actual_heat"].to_numpy(dtype=float)
    absolute_error = np.abs(error)
    rmse = math.sqrt(float(np.mean(error**2)))
    actual_mean = float(np.mean(actual))
    total_sum_of_squares = float(np.sum((actual - actual_mean) ** 2))
    residual_sum_of_squares = float(np.sum(error**2))
    return {
        "model": str(group["model"].iloc[0]),
        "metric_scope": scope,
        "n_forecast_starts": int(group["forecast_start"].nunique()),
        "n_rows": int(len(group)),
        "resolution": "hourly",
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
        "prediction_seconds_total": float(
            group.groupby("forecast_start", sort=False)["prediction_seconds"].first().sum()
        ),
        "prediction_seconds_mean_per_forecast_start": float(
            group.groupby("forecast_start", sort=False)["prediction_seconds"].first().mean()
        ),
    }


def calculate_summary_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, model_group in raw.groupby("model", sort=True):
        rows.append(calculate_group_metrics("all", model_group))
        rows.extend(
            calculate_group_metrics(f"selection:{selection}", group)
            for selection, group in model_group.groupby("selection", sort=True)
        )
        rows.extend(
            calculate_group_metrics(f"week:{week_id}", group)
            for week_id, group in model_group.groupby("week_id", sort=True)
        )
    return pd.DataFrame(rows)


def build_metadata(
    args: argparse.Namespace,
    run_id: str,
    selected_weeks: pd.DataFrame,
    starts: list[tuple[str, str, pd.Timestamp]],
    fit_seconds: float,
    total_seconds: float,
    models: list[str],
    weather_columns: list[str],
) -> dict[str, object]:
    metadata_doc = metadata_envelope(
        run_id=run_id,
        run_name=args.run_name,
        run_dir=args.output_dir / run_id,
        script_path=__file__,
        packages=AUTOGLUON_PACKAGES,
    )
    metadata_doc.update({
        "inputs": {
            "heat_path": str(args.heat_path),
            "heat_sha256": sha256_file(args.heat_path),
            "weather_path": str(args.weather_path),
            "weather_sha256": sha256_file(args.weather_path),
            "selected_weeks_path": str(args.selected_weeks_path),
            "selected_weeks_sha256": sha256_file(args.selected_weeks_path),
            "weather_columns": weather_columns,
        },
        "split": {
            "fit_target_period": "2023-01-01 <= timestamp < 2024-01-01",
            "prediction_policy": (
                "Predictor is fit on 2023 target values only. During 2024 prediction, "
                "observed heat history before each forecast start is provided as operational context."
            ),
        },
        "autogluon": {
            "prediction_length": args.prediction_hours,
            "freq": "h",
            "eval_metric": "RMSE",
            "time_limit_seconds": args.time_limit_seconds,
            "num_val_windows": args.num_val_windows,
            "enable_ensemble": not args.disable_ensemble,
            "include_neural_models": not args.disable_neural,
            "timestamp_axis": (
                "AutoGluon receives a synthetic continuous hourly model clock. "
                "Real Europe/Berlin timestamps are preserved in outputs."
            ),
            "hyperparameters": build_autogluon_hyperparameters_from_args(args),
            "models": models,
            "fit_seconds": fit_seconds,
        },
        "selected_forecast_weeks": selected_weeks.to_dict(orient="records"),
        "n_forecast_starts": len(starts),
        "total_seconds": total_seconds,
    })
    return metadata_doc


def main() -> None:
    args = parse_args()
    weather_columns = parse_weather_columns(args.weather_columns)
    total_start = time.perf_counter()

    selected_weeks = load_selected_weeks(args.selected_weeks_path)
    starts = forecast_starts_for_selected_weeks(
        selected_weeks,
        prediction_hours=args.prediction_hours,
        max_forecast_starts=args.max_forecast_starts,
    )

    print(f"Forecast starts requested: {len(starts):,}")
    print("Forecast windowing: non-overlapping 24h windows inside selected weeks")
    print(f"AutoGluon time limit seconds: {args.time_limit_seconds}")
    print(f"Weather columns: {', '.join(weather_columns)}")

    data = load_merged_data(args.heat_path, args.weather_path, weather_columns, HOURLY_STEP)
    validate_train_test_boundary(
        data,
        train_year=2023,
        test_year=2024,
        step=HOURLY_STEP,
        value_columns=["heat", *weather_columns],
    )
    data = add_synthetic_model_clock(data, HOURLY_STEP)
    train_data = data_slice_by_year(data, 2023)

    if args.dry_run:
        validate_dry_run(
            data=data,
            train_data=train_data,
            starts=starts,
            prediction_hours=args.prediction_hours,
            weather_columns=weather_columns,
        )
        total_seconds = time.perf_counter() - total_start
        print(f"Dry run completed successfully in {total_seconds:.2f}s.")
        return

    run_id = make_run_id(args.run_id, args.run_name)
    run_dir = args.output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    command_path = run_dir / "command.txt"
    command_path.write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    raw_output_path = run_dir / "raw_predictions.csv"
    metrics_output_path = run_dir / "metrics_per_forecast_start.csv"
    summary_output_path = run_dir / "metrics_summary.csv"
    leaderboard_output_path = run_dir / "autogluon_leaderboard.csv"
    validation_leaderboard_output_path = run_dir / "autogluon_validation_leaderboard.csv"
    metadata_path = run_dir / "run_metadata.json"

    print(f"Run directory: {run_dir}")

    train_frame = to_autogluon_frame(train_data, weather_columns, include_target=True)
    train_ts = to_ts_dataframe(train_frame)

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

    validation_leaderboard = predictor.leaderboard(display=False)
    validation_leaderboard.to_csv(validation_leaderboard_output_path, index=False)

    models = list(predictor.model_names())
    print(f"Models available for prediction: {', '.join(models)}")
    if not any("GBM" in model or "LightGBM" in model for model in models):
        raise RuntimeError(f"Expected a LightGBM/GBM model in AutoGluon models, got: {models}")
    if not any("XGB" in model or "XGBoost" in model for model in models):
        raise RuntimeError(f"Expected an XGBoost/XGB model in AutoGluon models, got: {models}")

    raw_forecasts = []
    metric_rows = []
    for start_number, (selection, week_id, forecast_start) in enumerate(starts, start=1):
        print(f"[{start_number}/{len(starts)}] Forecast start: {selection} {week_id} {forecast_start}")
        history_frame, future_covariates, actual = build_autogluon_prediction_inputs(
            data=data,
            forecast_start=forecast_start,
            prediction_steps=args.prediction_hours,
            history_start=pd.Timestamp("2023-01-01", tz=TIMEZONE),
            evaluation_year=2024,
            weather_columns=weather_columns,
            step=HOURLY_STEP,
        )
        history_ts = to_ts_dataframe(history_frame)
        future_covariates_ts = to_ts_dataframe(future_covariates)

        for model in models:
            prediction_start = time.perf_counter()
            prediction = predictor.predict(history_ts, known_covariates=future_covariates_ts, model=model)
            prediction_seconds = time.perf_counter() - prediction_start
            predicted = prediction_to_frame(prediction, model_name=model)
            forecast = build_forecast_output(
                predicted=predicted,
                actual=actual,
                selection=selection,
                week_id=week_id,
                forecast_start=forecast_start,
                prediction_seconds=prediction_seconds,
            )
            raw_forecasts.append(forecast)
            metrics = calculate_metrics(forecast)
            metric_rows.append(metrics)
            print(
                f"  {model}: RMSE={metrics['RMSE']:.3f}, "
                f"CVRMSE={metrics['CVRMSE_percent']:.2f}%, "
                f"prediction={prediction_seconds:.2f}s"
            )

    raw = pd.concat(raw_forecasts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = calculate_summary_metrics(raw)
    leaderboard = (
        summary[summary["metric_scope"] == "all"]
        .sort_values(["RMSE", "CVRMSE_percent"], ascending=True)
        .reset_index(drop=True)
    )
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))

    raw.to_csv(raw_output_path, index=False)
    metrics.to_csv(metrics_output_path, index=False)
    summary.to_csv(summary_output_path, index=False)
    leaderboard.to_csv(leaderboard_output_path, index=False)

    total_seconds = time.perf_counter() - total_start
    write_metadata(
        metadata_path,
        build_metadata(
            args=args,
            run_id=run_id,
            selected_weeks=selected_weeks,
            starts=starts,
            fit_seconds=fit_seconds,
            total_seconds=total_seconds,
            models=models,
            weather_columns=weather_columns,
        ),
    )

    print(f"Saved raw predictions: {raw_output_path}")
    print(f"Saved per-start metrics: {metrics_output_path}")
    print(f"Saved summary metrics: {summary_output_path}")
    print(f"Saved leaderboard: {leaderboard_output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Total seconds: {total_seconds:.2f}")


if __name__ == "__main__":
    main()
