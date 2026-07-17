from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from full_year_forecasting_utils import (
    HOURLY_STEP,
    QUARTER_STEP,
    TIMEZONE,
    UnsupportedDataCadenceError,
    load_merged_data,
    make_model_frame,
    validate_regular_rows,
)

DEFAULT_HEAT_PATH = Path("flensburg/demand/heat/heat_dh.csv")
DEFAULT_WEATHER_PATH = Path("flensburg/weather/flensburg_weather_temperature.csv")
DEFAULT_FORECAST_START = "2024-01-15 00:00:00+01:00"
DEFAULT_CONTEXT_DAYS = 365
DEFAULT_PREDICTION_HOURS = 24
DEFAULT_CONTEXT_PLOT_HOURS = 48
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_INTERVAL = "std1"
WEATHER_COLUMNS = ["temperature"]
RESOLUTION_TO_FREQ = {"quarter": "15min", "hourly": "1h"}
STEPS_PER_HOUR_BY_RESOLUTION = {"quarter": 4, "hourly": 1}
EXPECTED_STEP_BY_RESOLUTION = {
    "quarter": QUARTER_STEP,
    "hourly": HOURLY_STEP,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one native TabPFN-TS 24h heat-demand backtest with weather "
            "covariates and one year of context."
        )
    )
    parser.add_argument("--heat-path", type=Path, default=DEFAULT_HEAT_PATH)
    parser.add_argument("--weather-path", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--forecast-start", default=DEFAULT_FORECAST_START)
    parser.add_argument("--context-days", type=int, default=DEFAULT_CONTEXT_DAYS)
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
        "--context-plot-hours",
        type=int,
        default=DEFAULT_CONTEXT_PLOT_HOURS,
        help="Hours of historical context to include in the plot.",
    )
    parser.add_argument(
        "--mode",
        choices=("CLIENT", "LOCAL"),
        default="LOCAL",
        help="TabPFN-TS inference mode.",
    )
    parser.add_argument(
        "--interval",
        choices=("q10-q90", "std1", "std2"),
        default=DEFAULT_INTERVAL,
        help=(
            "Prediction interval to request and plot. std1 uses the normal-equivalent "
            "15.87%%-84.13%% quantiles; std2 uses 2.28%%-97.72%%."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and forecast windows without loading TabPFN-TS or writing outputs.",
    )
    return parser.parse_args()


def interval_quantiles(interval: str) -> tuple[float, float, float]:
    if interval == "q10-q90":
        return 0.1, 0.5, 0.9
    if interval == "std1":
        return 0.1587, 0.5, 0.8413
    if interval == "std2":
        return 0.0228, 0.5, 0.9772
    raise ValueError(f"Unsupported interval: {interval}")


def format_quantile_label(value: float) -> str:
    return f"q{value * 100:.2f}".rstrip("0").rstrip(".")


def prediction_steps(prediction_hours: int, resolution: str) -> int:
    if prediction_hours <= 0:
        raise ValueError(f"prediction-hours must be positive, got {prediction_hours}.")
    return prediction_hours * STEPS_PER_HOUR_BY_RESOLUTION[resolution]


def load_data(heat_path: Path, weather_path: Path, resolution: str) -> pd.DataFrame:
    if resolution not in STEPS_PER_HOUR_BY_RESOLUTION:
        raise ValueError(f"Unsupported resolution: {resolution}")
    step = EXPECTED_STEP_BY_RESOLUTION[resolution]
    try:
        df = load_merged_data(heat_path, weather_path, WEATHER_COLUMNS, step)
    except UnsupportedDataCadenceError as exc:
        raise SystemExit(str(exc)) from None
    print(f"Loaded {len(df):,} aligned {resolution} rows")
    return df


def infer_step_size(df: pd.DataFrame, resolution: str) -> pd.Timedelta:
    expected_step = EXPECTED_STEP_BY_RESOLUTION[resolution]
    validate_regular_rows(df, expected_step, f"{resolution} forecast data")
    return expected_step


def make_tabpfn_frame(df: pd.DataFrame) -> pd.DataFrame:
    return make_model_frame(
        df,
        target_column="heat",
        covariate_columns=WEATHER_COLUMNS,
        include_target=True,
    )


def build_windows(
    df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    context_hours: int,
    prediction_hours: int,
    resolution: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timedelta]:
    step = infer_step_size(df, resolution)
    n_prediction_steps = prediction_steps(prediction_hours, resolution)
    if context_hours <= 0:
        raise ValueError(f"context-hours must be positive, got {context_hours}.")
    context_start = forecast_start - pd.Timedelta(hours=context_hours)
    forecast_end_exclusive = forecast_start + pd.Timedelta(hours=prediction_hours)

    context_raw = df[(df["timestamp"] >= context_start) & (df["timestamp"] < forecast_start)]
    test_raw = df[
        (df["timestamp"] >= forecast_start)
        & (df["timestamp"] < forecast_end_exclusive)
    ]

    if context_raw.empty:
        raise ValueError("Context window is empty.")
    if len(test_raw) != n_prediction_steps:
        raise ValueError(
            "Forecast/test window does not contain the requested number of rows: "
            f"expected {n_prediction_steps}, got {len(test_raw)}."
        )

    min_context_start = df["timestamp"].min()
    if context_raw["timestamp"].min() > context_start:
        raise ValueError(
            f"Not enough context. Need data from {context_start}, "
            f"but dataset starts at {min_context_start}."
        )

    tabpfn_context = make_tabpfn_frame(context_raw)
    future_df = make_tabpfn_frame(test_raw).drop(columns=["target"])
    test_df = test_raw.rename(columns={"heat": "actual_heat"})[
        ["timestamp", "actual_heat", *WEATHER_COLUMNS]
    ].copy()
    test_df["timestamp"] = test_df["timestamp"].dt.tz_localize(None)

    return tabpfn_context, future_df, test_df, step


def validate_model_inputs(context_df: pd.DataFrame, future_df: pd.DataFrame) -> None:
    if "target" not in context_df.columns:
        raise ValueError("context_df must contain target.")
    if "target" in future_df.columns:
        raise ValueError("future_df must not contain target.")
    if context_df["target"].isna().any():
        raise ValueError("context_df contains missing target values.")
    if future_df[WEATHER_COLUMNS].isna().any().any():
        raise ValueError("future_df contains missing weather covariates.")
    if context_df["item_id"].nunique() != 1 or future_df["item_id"].nunique() != 1:
        raise ValueError("Expected exactly one item_id in context_df and future_df.")


def run_tabpfn_prediction(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame,
    mode: str,
    quantiles: list[float],
    max_context_length: int,
) -> pd.DataFrame:
    try:
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    tabpfn_mode = getattr(TabPFNMode, mode)
    pipeline = TabPFNTSPipeline(
        tabpfn_mode=tabpfn_mode,
        max_context_length=max_context_length,
    )
    if getattr(pipeline, "max_context_length", 0) < len(context_df):
        raise ValueError(
            "TabPFN-TS would truncate the context: "
            f"max_context_length={pipeline.max_context_length}, context rows={len(context_df)}."
        )
    return pipeline.predict_df(
        context_df=context_df,
        future_df=future_df,
        quantiles=quantiles,
    )


def find_column_by_numeric_value(columns: pd.Index, value: float) -> object | None:
    for column in columns:
        try:
            if math.isclose(float(column), value, rel_tol=0.0, abs_tol=1e-12):
                return column
        except (TypeError, ValueError):
            continue
    return None


def flatten_predictions(
    pred_df: pd.DataFrame,
    lower_quantile: float,
    median_quantile: float,
    upper_quantile: float,
) -> pd.DataFrame:
    out = pred_df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    elif "timestamp" not in out.columns:
        out = out.reset_index()

    if "timestamp" not in out.columns:
        raise ValueError(f"Prediction output has no timestamp column: {out.columns.tolist()}")
    if "target" not in out.columns:
        raise ValueError(f"Prediction output has no target column: {out.columns.tolist()}")

    out = out.rename(columns={"target": "predicted_heat"})
    for value, target in (
        (lower_quantile, "q_lower"),
        (median_quantile, "q50"),
        (upper_quantile, "q_upper"),
    ):
        source = find_column_by_numeric_value(out.columns, value)
        if source is not None:
            out = out.rename(columns={source: target})

    if "q50" not in out.columns:
        out["q50"] = out["predicted_heat"]
    for col in ("q_lower", "q_upper"):
        if col not in out.columns:
            out[col] = np.nan

    return out[["timestamp", "predicted_heat", "q_lower", "q50", "q_upper"]].copy()


def build_forecast_output(
    pred_df: pd.DataFrame,
    test_df: pd.DataFrame,
    step: pd.Timedelta,
    lower_quantile: float,
    median_quantile: float,
    upper_quantile: float,
) -> pd.DataFrame:
    pred = flatten_predictions(pred_df, lower_quantile, median_quantile, upper_quantile)
    out = test_df.merge(pred, on="timestamp", how="left", validate="one_to_one")

    if out["predicted_heat"].isna().any():
        raise ValueError("Some forecast timestamps did not receive predictions.")

    out.insert(0, "horizon_step", np.arange(1, len(out) + 1))
    out.insert(1, "horizon_minutes", out["horizon_step"] * int(step.total_seconds() / 60))
    out.insert(2, "lower_quantile", lower_quantile)
    out.insert(3, "upper_quantile", upper_quantile)
    out["error"] = out["predicted_heat"] - out["actual_heat"]
    out["absolute_error"] = out["error"].abs()

    denominator = out["actual_heat"].abs() + out["predicted_heat"].abs()
    out["sape"] = np.where(denominator > 0, 2 * out["absolute_error"] / denominator, np.nan)
    return out


def calculate_metrics(
    forecast: pd.DataFrame,
    args: argparse.Namespace,
    timings: dict[str, float],
    n_context_rows: int,
) -> pd.DataFrame:
    error = forecast["error"].to_numpy()
    actual = forecast["actual_heat"].to_numpy()
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
    cvrmse_percent = float(rmse / mean_actual * 100) if mean_actual != 0 else np.nan

    metrics = {
        "forecast_start": args.forecast_start,
        "resolution": args.resolution,
        "context_days": args.context_days,
        "prediction_hours": args.prediction_hours,
        "prediction_steps": prediction_steps(args.prediction_hours, args.resolution),
        "mode": args.mode,
        "interval": args.interval,
        "n_context_rows": n_context_rows,
        "n_forecast_rows": len(forecast),
        "MAE": float(np.mean(absolute_error)),
        "RMSE": rmse,
        "R2": r2_score,
        "CVRMSE_percent": cvrmse_percent,
        "sMAPE_percent": smape,
        "mean_error": float(np.mean(error)),
        "max_absolute_error": float(np.max(absolute_error)),
        **timings,
    }
    return pd.DataFrame([metrics])


def plot_forecast(
    context_df: pd.DataFrame,
    forecast: pd.DataFrame,
    forecast_start: pd.Timestamp,
    context_plot_hours: int,
    output_path: Path,
    metrics: pd.DataFrame,
) -> None:
    if forecast_start.tzinfo is not None:
        forecast_start = forecast_start.tz_localize(None)

    plot_start = forecast_start - pd.Timedelta(hours=context_plot_hours)
    context_plot = context_df[context_df["timestamp"] >= plot_start]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        context_plot["timestamp"],
        context_plot["target"],
        label=f"Context last {context_plot_hours}h",
        color="tab:blue",
        linewidth=1.5,
    )
    ax.plot(
        forecast["timestamp"],
        forecast["actual_heat"],
        label="Actual heat",
        color="black",
        linewidth=1.8,
    )
    ax.plot(
        forecast["timestamp"],
        forecast["predicted_heat"],
        label="TabPFN-TS prediction",
        color="tab:orange",
        linewidth=1.8,
    )

    has_interval = forecast[["q_lower", "q_upper"]].notna().all().all()
    if has_interval:
        ax.fill_between(
            forecast["timestamp"],
            forecast["q_lower"],
            forecast["q_upper"],
            color="tab:orange",
            alpha=0.2,
            label=(
                f"{format_quantile_label(float(forecast['lower_quantile'].iloc[0]))}-"
                f"{format_quantile_label(float(forecast['upper_quantile'].iloc[0]))} interval"
            ),
        )
    else:
        print("Warning: interval quantiles are missing; forecast interval will not be plotted.")

    ax.axvline(forecast_start, color="grey", linestyle="--", linewidth=1)
    mae = metrics.loc[0, "MAE"]
    rmse = metrics.loc[0, "RMSE"]
    r2_score = metrics.loc[0, "R2"]
    cvrmse = metrics.loc[0, "CVRMSE_percent"]
    pred_seconds = metrics.loc[0, "prediction_seconds"]
    ax.set_title(
        f"TabPFN-TS heat forecast from {forecast_start} | "
        f"MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2_score:.3f}, "
        f"CVRMSE={cvrmse:.2f}%, prediction={pred_seconds:.1f}s"
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Heat demand")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()

    load_start = time.perf_counter()
    data = load_data(args.heat_path, args.weather_path, resolution=args.resolution)
    data_loading_seconds = time.perf_counter() - load_start

    forecast_start = pd.Timestamp(args.forecast_start)
    if forecast_start.tzinfo is None:
        forecast_start = forecast_start.tz_localize(TIMEZONE)
    else:
        forecast_start = forecast_start.tz_convert(TIMEZONE)
    context_df, future_df, test_df, step = build_windows(
        data,
        forecast_start=forecast_start,
        context_hours=args.context_days * 24,
        prediction_hours=args.prediction_hours,
        resolution=args.resolution,
    )
    validate_model_inputs(context_df, future_df)

    print(f"Forecast start: {forecast_start}")
    print(f"Resolution: {args.resolution}")
    print(f"Prediction hours: {args.prediction_hours}")
    print(f"Context rows: {len(context_df):,}")
    print(f"Forecast rows: {len(future_df):,}")
    if args.dry_run:
        print("Dry run complete. No model loaded and no outputs written.")
        return
    print(f"Running TabPFN-TS in {args.mode} mode...")

    prediction_start = time.perf_counter()
    lower_quantile, median_quantile, upper_quantile = interval_quantiles(args.interval)
    pred_df = run_tabpfn_prediction(
        context_df,
        future_df,
        args.mode,
        quantiles=[lower_quantile, median_quantile, upper_quantile],
        max_context_length=len(context_df),
    )
    prediction_seconds = time.perf_counter() - prediction_start

    forecast = build_forecast_output(
        pred_df,
        test_df,
        step,
        lower_quantile,
        median_quantile,
        upper_quantile,
    )
    total_seconds = time.perf_counter() - total_start
    timings = {
        "data_loading_seconds": data_loading_seconds,
        "prediction_seconds": prediction_seconds,
        "total_seconds": total_seconds,
    }
    metrics = calculate_metrics(forecast, args, timings, len(context_df))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    date_tag = forecast_start.strftime("%Y-%m-%d")
    forecast_path = args.output_dir / f"tabpfn_ts_heat_forecast_{date_tag}.csv"
    metrics_path = args.output_dir / f"tabpfn_ts_heat_forecast_{date_tag}_metrics.csv"
    plot_path = args.output_dir / f"tabpfn_ts_heat_forecast_{date_tag}.png"

    forecast.to_csv(forecast_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    plot_forecast(
        context_df=context_df,
        forecast=forecast,
        forecast_start=forecast_start,
        context_plot_hours=args.context_plot_hours,
        output_path=plot_path,
        metrics=metrics,
    )

    print(f"Prediction seconds: {prediction_seconds:.2f}")
    print(f"MAE: {metrics.loc[0, 'MAE']:.3f}")
    print(f"RMSE: {metrics.loc[0, 'RMSE']:.3f}")
    print(f"R2: {metrics.loc[0, 'R2']:.3f}")
    print(f"CVRMSE: {metrics.loc[0, 'CVRMSE_percent']:.3f}%")
    print(f"sMAPE: {metrics.loc[0, 'sMAPE_percent']:.3f}%")
    print(f"Saved forecast CSV: {forecast_path}")
    print(f"Saved metrics CSV: {metrics_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
