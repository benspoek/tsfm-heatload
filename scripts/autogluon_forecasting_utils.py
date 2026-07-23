from __future__ import annotations

import pandas as pd

from full_year_forecasting_utils import make_model_frame


def to_autogluon_frame(
    data: pd.DataFrame,
    weather_columns: list[str],
    include_target: bool,
) -> pd.DataFrame:
    if "model_timestamp" not in data.columns:
        raise ValueError("AutoGluon data must contain a synthetic model_timestamp column.")
    return make_model_frame(data, "heat", weather_columns, include_target)


def _assert_exact_index(actual: pd.Series, expected: pd.DatetimeIndex, label: str) -> None:
    actual_index = pd.DatetimeIndex(actual)
    if not actual_index.equals(expected):
        missing = expected.difference(actual_index)[:5].tolist()
        unexpected = actual_index.difference(expected)[:5].tolist()
        raise ValueError(
            f"{label} is not exactly aligned. "
            f"Missing examples: {missing}; unexpected examples: {unexpected}"
        )


def build_autogluon_prediction_inputs(
    data: pd.DataFrame,
    forecast_start: pd.Timestamp,
    prediction_steps: int,
    history_start: pd.Timestamp,
    evaluation_year: int,
    weather_columns: list[str],
    step: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if prediction_steps <= 0:
        raise ValueError(f"prediction steps must be positive, got {prediction_steps}.")
    if step <= pd.Timedelta(0):
        raise ValueError(f"forecast step must be positive, got {step}.")
    if forecast_start.tzinfo is None or history_start.tzinfo is None:
        raise ValueError("forecast_start and history_start must be timezone-aware.")

    required = {"timestamp", "model_timestamp", "heat", *weather_columns}
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise ValueError(f"AutoGluon source data is missing columns: {missing_columns}")

    forecast_end = forecast_start + prediction_steps * step
    history = data[
        (data["timestamp"] >= history_start) & (data["timestamp"] < forecast_start)
    ].sort_values("timestamp").reset_index(drop=True)
    future = data[
        (data["timestamp"] >= forecast_start) & (data["timestamp"] < forecast_end)
    ].sort_values("timestamp").reset_index(drop=True)
    if history.empty:
        raise ValueError(f"History is empty before forecast start {forecast_start}.")

    expected_history_end = forecast_start - step
    actual_history_end = history["timestamp"].iloc[-1]
    if actual_history_end != expected_history_end:
        raise ValueError(
            f"Target history for {forecast_start} ends at {actual_history_end}, "
            f"expected {expected_history_end}."
        )
    if not history["timestamp"].lt(forecast_start).all():
        raise ValueError(f"Target history leaks into the forecast horizon at {forecast_start}.")

    history_required = ["heat", *weather_columns]
    if history[history_required].isna().any().any():
        missing_values = history[history_required].isna().sum().to_dict()
        raise ValueError(f"Target history contains missing values: {missing_values}")
    future_required = ["heat", *weather_columns]
    if future[future_required].isna().any().any():
        missing_values = future[future_required].isna().sum().to_dict()
        raise ValueError(f"Forecast horizon contains missing values: {missing_values}")

    expected_real_horizon = pd.date_range(
        start=forecast_start,
        periods=prediction_steps,
        freq=step,
    )
    _assert_exact_index(future["timestamp"], expected_real_horizon, "Real forecast horizon")
    if future["timestamp"].dt.year.ne(evaluation_year).any():
        raise ValueError(
            f"Evaluation horizon includes timestamps outside {evaluation_year} at {forecast_start}."
        )

    history_model_index = pd.DatetimeIndex(history["model_timestamp"])
    expected_history_model_index = pd.date_range(
        start=history_model_index[0],
        periods=len(history_model_index),
        freq=step,
    )
    if not history_model_index.equals(expected_history_model_index):
        raise ValueError("Synthetic model clock is not regular in target history.")
    expected_model_horizon = pd.date_range(
        start=history_model_index[-1] + step,
        periods=prediction_steps,
        freq=step,
    )
    _assert_exact_index(future["model_timestamp"], expected_model_horizon, "Synthetic forecast horizon")

    history_frame = to_autogluon_frame(history, weather_columns, include_target=True)
    future_covariates = to_autogluon_frame(future, weather_columns, include_target=False)
    if "target" in future_covariates.columns:
        raise ValueError("AutoGluon known covariates must not contain target.")
    _assert_exact_index(
        future_covariates["timestamp"],
        expected_model_horizon,
        "AutoGluon known covariates",
    )
    if history_frame["timestamp"].iloc[-1] >= future_covariates["timestamp"].iloc[0]:
        raise ValueError("AutoGluon target history overlaps the known-covariate horizon.")

    actual = future[["timestamp", "model_timestamp", "heat", *weather_columns]].rename(
        columns={"heat": "actual_heat"}
    ).copy()
    _assert_exact_index(actual["model_timestamp"], expected_model_horizon, "Actual forecast horizon")
    return history_frame, future_covariates, actual


def prediction_to_frame(prediction, model_name: str) -> pd.DataFrame:
    pred = prediction.reset_index()
    if "mean" in pred.columns:
        value_column = "mean"
    elif "0.5" in pred.columns:
        value_column = "0.5"
    else:
        numeric_columns = [column for column in pred.columns if column not in {"item_id", "timestamp"}]
        if not numeric_columns:
            raise ValueError(f"No prediction columns found for model {model_name}: {pred.columns.tolist()}")
        value_column = numeric_columns[0]

    out = pred[["timestamp", value_column]].rename(
        columns={"timestamp": "model_timestamp", value_column: "predicted_heat"}
    ).copy()
    out["model"] = model_name
    for column in pred.columns:
        if column in {"item_id", "timestamp", value_column, "mean"}:
            continue
        try:
            quantile = float(column)
        except (TypeError, ValueError):
            continue
        label = f"q{quantile * 100:.2f}".rstrip("0").rstrip(".").replace(".", "_")
        out[label] = pred[column]
    return out


def align_prediction_with_actual(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    forecast_start: pd.Timestamp,
) -> pd.DataFrame:
    for label, frame in (("prediction", predicted), ("actual", actual)):
        if "model_timestamp" not in frame.columns:
            raise ValueError(f"AutoGluon {label} is missing model_timestamp.")
        if frame["model_timestamp"].duplicated().any():
            raise ValueError(f"AutoGluon {label} contains duplicate model timestamps.")
    expected = pd.DatetimeIndex(actual["model_timestamp"])
    _assert_exact_index(predicted["model_timestamp"], expected, "AutoGluon prediction horizon")

    out = actual.merge(predicted, on="model_timestamp", how="left", validate="one_to_one")
    if out["predicted_heat"].isna().any():
        model = predicted["model"].iloc[0] if not predicted.empty and "model" in predicted else "unknown"
        raise ValueError(f"Missing predictions for {model} at {forecast_start}.")
    return out
