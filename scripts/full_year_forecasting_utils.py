from __future__ import annotations

import hashlib
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


TIMEZONE = "Europe/Berlin"
ITEM_ID = "heat_dh"
QUARTER_STEP = pd.Timedelta(minutes=15)
HOURLY_STEP = pd.Timedelta(hours=1)
STEPS_PER_HOUR = 4
DEFAULT_SELECTED_WEEKS_PATH = Path("flensburg/weather/representative_weeks_2024.csv")
REQUIRED_SELECTED_WEEK_LABELS = {
    "hottest",
    "coldest",
    "highest_temperature_fluctuation",
}


class UnsupportedDataCadenceError(ValueError):
    """Raised when an experiment needs finer source data than are available."""


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_metadata() -> dict[str, object]:
    status = run_command(["git", "status", "--porcelain"])
    return {
        "commit": run_command(["git", "rev-parse", "HEAD"]),
        "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
    }


def parse_timestamp_series(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True, errors="raise")
    return timestamps.dt.tz_convert(TIMEZONE)


def validate_timeseries_frame(
    df: pd.DataFrame,
    value_columns: list[str],
    label: str,
    expected_step: pd.Timedelta | None = None,
) -> pd.Timedelta:
    required = {"timestamp", *value_columns}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"{label} is missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError(f"{label} is empty.")
    if df["timestamp"].isna().any():
        raise ValueError(f"{label} contains missing timestamps.")
    if df["timestamp"].dt.tz is None:
        raise ValueError(f"{label} timestamps must be timezone-aware.")
    if df["timestamp"].duplicated().any():
        duplicated = df.loc[df["timestamp"].duplicated(), "timestamp"].head().tolist()
        raise ValueError(f"{label} contains duplicate timestamps, examples: {duplicated}")
    if df[value_columns].isna().any().any():
        missing_values = df[value_columns].isna().sum().to_dict()
        raise ValueError(f"{label} contains missing values: {missing_values}")

    source_step = infer_native_step(df, label)
    if expected_step is not None and source_step != expected_step:
        raise ValueError(f"{label} has cadence {source_step}, expected {expected_step}.")
    validate_regular_rows(df, source_step, label)
    return source_step


def read_heat_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Heat file not found: {path}")
    heat = pd.read_csv(path, index_col=0)
    if "heat" not in heat.columns:
        raise ValueError(f"Expected column 'heat' in {path}")
    heat.index = pd.to_datetime(heat.index, utc=True, errors="raise").tz_convert(TIMEZONE)
    heat.index.name = "timestamp"
    heat = heat.reset_index()[["timestamp", "heat"]].sort_values("timestamp").reset_index(drop=True)
    validate_timeseries_frame(heat, ["heat"], f"heat data ({path})")
    return heat


def read_weather_csv(path: Path, weather_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Weather file not found: {path}")
    weather = pd.read_csv(path)
    timestamp_column = "date" if "date" in weather.columns else weather.columns[0]
    missing = sorted(set(weather_columns) - set(weather.columns))
    if missing:
        raise ValueError(f"Weather file is missing columns: {missing}")
    weather["timestamp"] = parse_timestamp_series(weather[timestamp_column])
    weather = (
        weather[["timestamp", *weather_columns]]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    validate_timeseries_frame(weather, weather_columns, f"weather data ({path})")
    return weather


def read_weather_comparison_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Weather comparison file not found: {path}")
    weather = pd.read_csv(path)
    required = {"date", "temperature_observed", "temperature_forecast_24h"}
    missing = sorted(required - set(weather.columns))
    if missing:
        raise ValueError(f"Weather comparison file is missing columns: {missing}")
    weather["timestamp"] = parse_timestamp_series(weather["date"])
    weather = (
        weather[["timestamp", "temperature_observed", "temperature_forecast_24h"]]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    # Missing archived forecasts are allowed here and checked for each requested window.
    validate_timeseries_frame(
        weather,
        ["temperature_observed"],
        f"weather comparison data ({path})",
        expected_step=HOURLY_STEP,
    )
    return weather


def infer_native_step(df: pd.DataFrame, label: str) -> pd.Timedelta:
    timestamps = df["timestamp"].sort_values()
    deltas = timestamps.diff().dropna()
    if deltas.empty:
        raise ValueError(f"{label} has fewer than two timestamps.")
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        raise ValueError(f"{label} has no increasing timestamp differences.")
    return positive.mode().iloc[0]


def cadence_name(step: pd.Timedelta) -> str:
    if step == QUARTER_STEP:
        return "15-minute"
    if step == HOURLY_STEP:
        return "hourly"
    minutes = step.total_seconds() / 60
    if minutes.is_integer():
        return f"{int(minutes)}-minute"
    return str(step)


def require_source_cadence(
    df: pd.DataFrame,
    maximum_step: pd.Timedelta,
    label: str,
    purpose: str,
) -> pd.Timedelta:
    source_step = infer_native_step(df, label)
    if source_step > maximum_step:
        raise UnsupportedDataCadenceError(
            f"Unsupported cadence for {purpose}: {label} is {cadence_name(source_step)}, "
            f"but native {cadence_name(maximum_step)} or finer data is required. "
            "Coarser data cannot supply the missing target variation."
        )
    return source_step


def expected_rows_per_target_step(source_step: pd.Timedelta, target_step: pd.Timedelta, label: str) -> int:
    source_seconds = source_step.total_seconds()
    target_seconds = target_step.total_seconds()
    if source_seconds <= 0:
        raise ValueError(f"{label} has invalid source step: {source_step}")
    if source_seconds > target_seconds:
        raise UnsupportedDataCadenceError(
            f"Unsupported cadence: {label} is {cadence_name(source_step)}, but "
            f"{cadence_name(target_step)} output was requested. Coarser data cannot be upsampled."
        )
    ratio = target_seconds / source_seconds
    rounded = round(ratio)
    if not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{label} source step {source_step} does not divide target step {target_step} cleanly."
        )
    return int(rounded)


def aggregate_to_step(
    df: pd.DataFrame,
    value_columns: list[str],
    step: pd.Timedelta,
    label: str,
) -> pd.DataFrame:
    source_step = validate_timeseries_frame(df, value_columns, label)
    expected_count = expected_rows_per_target_step(source_step, step, label)
    if step == QUARTER_STEP:
        if expected_count != 1:
            raise ValueError(f"{label} cannot be upsampled from {source_step} to 15 minutes.")
        out = df[["timestamp", *value_columns]].copy().sort_values("timestamp").reset_index(drop=True)
    elif step == HOURLY_STEP:
        grouped = df.set_index("timestamp").resample("1h")
        means = grouped[value_columns].mean()
        counts = grouped[value_columns].count()
        complete = counts.eq(expected_count).all(axis=1)
        out = means.loc[complete].reset_index()
    else:
        raise ValueError(f"Unsupported aggregation step: {step}")

    if out["timestamp"].duplicated().any():
        duplicated = out.loc[out["timestamp"].duplicated(), "timestamp"].head().tolist()
        raise ValueError(f"Duplicate timestamps after aggregation, examples: {duplicated}")
    if out[value_columns].isna().any().any():
        missing_counts = out[value_columns].isna().sum().to_dict()
        raise ValueError(f"Missing values after aggregation: {missing_counts}")
    return out


def align_timeseries_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_value_columns: list[str],
    right_value_columns: list[str],
    step: pd.Timedelta,
    label: str,
) -> pd.DataFrame:
    validate_timeseries_frame(left, left_value_columns, f"{label} left input", expected_step=step)
    validate_timeseries_frame(right, right_value_columns, f"{label} right input", expected_step=step)

    overlap_start = max(left["timestamp"].min(), right["timestamp"].min())
    overlap_end = min(left["timestamp"].max(), right["timestamp"].max())
    if overlap_start > overlap_end:
        raise ValueError(f"{label} inputs do not overlap.")
    left_overlap = left[left["timestamp"].between(overlap_start, overlap_end)]
    right_overlap = right[right["timestamp"].between(overlap_start, overlap_end)]
    left_index = pd.DatetimeIndex(left_overlap["timestamp"])
    right_index = pd.DatetimeIndex(right_overlap["timestamp"])
    if not left_index.equals(right_index):
        missing_right = left_index.difference(right_index)[:5].tolist()
        missing_left = right_index.difference(left_index)[:5].tolist()
        raise ValueError(
            f"{label} timestamps are not aligned inside their common coverage. "
            f"Missing right-side examples: {missing_right}; missing left-side examples: {missing_left}"
        )

    data = left_overlap.merge(right_overlap, on="timestamp", how="inner", validate="one_to_one")
    data = data.sort_values("timestamp").reset_index(drop=True)
    validate_timeseries_frame(
        data,
        [*left_value_columns, *right_value_columns],
        label,
        expected_step=step,
    )
    return data


def load_merged_data(
    heat_path: Path,
    weather_path: Path,
    weather_columns: list[str],
    step: pd.Timedelta,
    maximum_heat_source_step: pd.Timedelta | None = None,
    heat_cadence_purpose: str = "this experiment",
) -> pd.DataFrame:
    heat = read_heat_csv(heat_path)
    if maximum_heat_source_step is not None:
        require_source_cadence(
            heat,
            maximum_step=maximum_heat_source_step,
            label=f"heat data ({heat_path})",
            purpose=heat_cadence_purpose,
        )
    weather = read_weather_csv(weather_path, weather_columns)

    heat_agg = aggregate_to_step(heat, ["heat"], step, label="heat")
    weather_agg = aggregate_to_step(weather, weather_columns, step, label="weather")

    return align_timeseries_frames(
        heat_agg,
        weather_agg,
        left_value_columns=["heat"],
        right_value_columns=weather_columns,
        step=step,
        label="merged heat/weather data",
    )


def load_residual_multiresolution_data(
    heat_path: Path,
    weather_path: Path,
    weather_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quarter = load_merged_data(
        heat_path,
        weather_path,
        weather_columns,
        step=QUARTER_STEP,
        maximum_heat_source_step=QUARTER_STEP,
        heat_cadence_purpose="15-minute residual forecasting",
    )
    hourly = aggregate_to_step(
        quarter,
        ["heat", *weather_columns],
        HOURLY_STEP,
        label="stacked base data",
    )
    return quarter, hourly


def load_heat_weather_comparison_data(heat_path: Path, weather_path: Path) -> pd.DataFrame:
    heat = aggregate_to_step(read_heat_csv(heat_path), ["heat"], HOURLY_STEP, label="heat")
    weather = read_weather_comparison_csv(weather_path)
    return align_timeseries_frames(
        heat,
        weather,
        left_value_columns=["heat"],
        right_value_columns=["temperature_observed"],
        step=HOURLY_STEP,
        label="merged heat/weather comparison data",
    )


def to_naive_datetime(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="raise")
    if timestamps.dt.tz is None:
        return timestamps
    return timestamps.dt.tz_localize(None)


def add_synthetic_model_clock(data: pd.DataFrame, step: pd.Timedelta) -> pd.DataFrame:
    """Add a continuous naive timestamp axis for models that cannot represent DST transitions."""
    out = data.sort_values("timestamp").reset_index(drop=True).copy()
    seconds = step.total_seconds()
    if seconds <= 0:
        raise ValueError(f"Invalid model-clock step: {step}")
    start = pd.Timestamp("2000-01-01 00:00:00")
    out["model_timestamp"] = start + pd.to_timedelta(np.arange(len(out)) * seconds, unit="s")
    if out["model_timestamp"].duplicated().any():
        raise ValueError("Synthetic model timestamps contain duplicates.")
    return out


def validate_regular_rows(df: pd.DataFrame, step: pd.Timedelta, label: str) -> None:
    if step <= pd.Timedelta(0):
        raise ValueError(f"{label} has invalid expected step: {step}")
    timestamps = df["timestamp"].sort_values()
    if timestamps.duplicated().any():
        duplicated = timestamps[timestamps.duplicated()].head().tolist()
        raise ValueError(f"{label} contains duplicate timestamps: {duplicated}")
    deltas = timestamps.diff().dropna()
    if deltas.empty:
        raise ValueError(f"{label} has fewer than two rows.")
    if not deltas.eq(step).all():
        bad = deltas.loc[~deltas.eq(step)].head().tolist()
        raise ValueError(f"{label} is not regular at {step}: {bad}")


def data_slice_by_year(data: pd.DataFrame, year: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz=TIMEZONE)
    end = pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz=TIMEZONE)
    return data[(data["timestamp"] >= start) & (data["timestamp"] < end)].copy()


def validate_complete_year(
    data: pd.DataFrame,
    year: int,
    step: pd.Timedelta,
    value_columns: list[str],
    label: str,
) -> pd.DataFrame:
    year_data = data_slice_by_year(data, year).sort_values("timestamp").reset_index(drop=True)
    if year_data.empty:
        raise ValueError(f"{label} contains no rows for {year}.")
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz=TIMEZONE)
    end = pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz=TIMEZONE)
    expected = pd.date_range(start=start, end=end, freq=step, inclusive="left")
    actual = pd.DatetimeIndex(year_data["timestamp"])
    if not actual.equals(expected):
        missing = expected.difference(actual)[:5].tolist()
        unexpected = actual.difference(expected)[:5].tolist()
        raise ValueError(
            f"{label} does not contain a complete {year} at {step}. "
            f"Expected {len(expected)} rows, found {len(actual)}. "
            f"Missing examples: {missing}; unexpected examples: {unexpected}"
        )
    validate_timeseries_frame(year_data, value_columns, f"{label} {year}", expected_step=step)
    return year_data


def validate_train_test_boundary(
    data: pd.DataFrame,
    train_year: int,
    test_year: int,
    step: pd.Timedelta,
    value_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_year >= test_year:
        raise ValueError(f"train year {train_year} must be earlier than test year {test_year}.")
    train = validate_complete_year(data, train_year, step, value_columns, "training data")
    test = validate_complete_year(data, test_year, step, value_columns, "test data")
    if train["timestamp"].max() >= test["timestamp"].min():
        raise ValueError("Training and test timestamps overlap.")
    return train, test


def local_midnight(date_value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(date_value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(TIMEZONE)
    else:
        timestamp = timestamp.tz_convert(TIMEZONE)
    if any((timestamp.hour, timestamp.minute, timestamp.second, timestamp.microsecond)):
        raise ValueError(f"Selected week must start at local midnight: {date_value}")
    return timestamp


def load_selected_weeks(path: Path = DEFAULT_SELECTED_WEEKS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Selected weeks file not found: {path}. "
            "Regenerate it with scripts/select_flensburg_representative_weeks.py."
        )
    selected = pd.read_csv(path)
    required = {"selection", "week_id", "week_start"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Selected weeks file is missing columns: {missing}")
    if selected[list(required)].isna().any().any():
        raise ValueError("Selected weeks file contains missing selection fields.")
    if selected["selection"].duplicated().any():
        duplicates = selected.loc[selected["selection"].duplicated(), "selection"].tolist()
        raise ValueError(f"Selected weeks file contains duplicate selections: {duplicates}")
    labels = set(selected["selection"].astype(str))
    if labels != REQUIRED_SELECTED_WEEK_LABELS:
        raise ValueError(
            "Selected weeks must contain exactly these labels: "
            f"{sorted(REQUIRED_SELECTED_WEEK_LABELS)}; found {sorted(labels)}"
        )

    starts = []
    for row in selected.itertuples(index=False):
        start = local_midnight(row.week_start)
        if start.isoweekday() != 1:
            raise ValueError(f"Selected week {row.week_id} does not start on Monday: {start}")
        iso = start.isocalendar()
        expected_id = f"{iso.year}-W{iso.week:02d}"
        if str(row.week_id) != expected_id:
            raise ValueError(
                f"Selected week ID {row.week_id!r} does not match start {start.date()} ({expected_id})."
            )
        starts.append(start)
    out = selected.copy()
    out["week_start"] = starts
    return out


def forecast_starts_for_selected_weeks(
    selected_weeks: pd.DataFrame,
    prediction_hours: int,
    stride_hours: int | None = None,
    max_forecast_starts: int | None = None,
    allow_horizon_past_week: bool = False,
) -> list[tuple[str, str, pd.Timestamp]]:
    if prediction_hours <= 0:
        raise ValueError(f"prediction hours must be positive, got {prediction_hours}.")
    stride_hours = prediction_hours if stride_hours is None else stride_hours
    if stride_hours <= 0:
        raise ValueError(f"stride hours must be positive, got {stride_hours}.")
    if max_forecast_starts is not None and max_forecast_starts <= 0:
        raise ValueError(f"max forecast starts must be positive, got {max_forecast_starts}.")

    starts = []
    for row in selected_weeks.itertuples(index=False):
        week_start = local_midnight(row.week_start)
        week_end = week_start + pd.DateOffset(days=7)
        for forecast_start in forecast_starts_between(
            week_start,
            week_end,
            prediction_hours=prediction_hours,
            stride_hours=stride_hours,
            allow_horizon_past_end=allow_horizon_past_week,
        ):
            starts.append((str(row.selection), str(row.week_id), forecast_start))
    if max_forecast_starts is not None:
        starts = starts[:max_forecast_starts]
    return starts


def forecast_starts_between(
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    prediction_hours: int,
    stride_hours: int,
    allow_horizon_past_end: bool = False,
) -> list[pd.Timestamp]:
    if prediction_hours <= 0:
        raise ValueError(f"prediction hours must be positive, got {prediction_hours}.")
    if stride_hours <= 0:
        raise ValueError(f"stride hours must be positive, got {stride_hours}.")
    if start >= end_exclusive:
        raise ValueError(f"forecast interval must be increasing: {start} to {end_exclusive}.")
    horizon = pd.Timedelta(hours=prediction_hours)
    stride = pd.Timedelta(hours=stride_hours)
    starts = []
    forecast_start = start
    while forecast_start < end_exclusive:
        if not allow_horizon_past_end and forecast_start + horizon > end_exclusive:
            break
        starts.append(forecast_start)
        forecast_start += stride
    return starts


def forecast_starts_from_rows(
    data: pd.DataFrame,
    year: int,
    horizon_steps: int,
    stride_steps: int,
    value_columns: list[str] | None = None,
) -> pd.DataFrame:
    if horizon_steps <= 0:
        raise ValueError(f"horizon steps must be positive, got {horizon_steps}.")
    if stride_steps <= 0:
        raise ValueError(f"stride steps must be positive, got {stride_steps}.")
    step = infer_native_step(data, "forecast data")
    if value_columns is None:
        value_columns = [column for column in data.columns if column not in {"timestamp", "model_timestamp"}]
    year_data = validate_complete_year(data, year, step, value_columns, "forecast data")
    rows = []
    for start_pos in range(0, len(year_data), stride_steps):
        end_pos = start_pos + horizon_steps
        if end_pos > len(year_data):
            break
        rows.append(
            {
                "forecast_start": year_data.loc[start_pos, "timestamp"],
                "horizon_end": year_data.loc[end_pos - 1, "timestamp"],
                "start_pos": start_pos,
                "horizon_steps": horizon_steps,
            }
        )
    return pd.DataFrame(rows)


def make_model_frame(
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


def build_tabpfn_window(
    data: pd.DataFrame,
    forecast_start: pd.Timestamp,
    context_steps: int,
    horizon_steps: int,
    target_column: str,
    covariate_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    context_df = make_model_frame(context_raw, target_column, covariate_columns, include_target=True)
    future_df = make_model_frame(future_raw, target_column, covariate_columns, include_target=False)
    if "target" in future_df.columns:
        raise ValueError("future_df must not contain target.")
    actual_columns = ["timestamp"]
    if "model_timestamp" in future_raw.columns:
        actual_columns.append("model_timestamp")
    actual_columns.extend([target_column, *covariate_columns])
    actual = future_raw[actual_columns].rename(
        columns={target_column: "actual_heat"}
    )
    if "model_timestamp" in actual.columns:
        actual["model_timestamp"] = to_naive_datetime(actual["model_timestamp"])
    else:
        actual["timestamp"] = to_naive_datetime(actual["timestamp"])
    return context_df, future_df, actual


def find_column_by_numeric_value(columns: pd.Index, value: float) -> object | None:
    for column in columns:
        try:
            if math.isclose(float(column), value, rel_tol=0.0, abs_tol=1e-12):
                return column
        except (TypeError, ValueError):
            continue
    return None


def flatten_tabpfn_prediction(pred_df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    out = pred_df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    elif "timestamp" not in out.columns:
        out = out.reset_index()
    if "timestamp" not in out.columns or "target" not in out.columns:
        raise ValueError(f"Unexpected prediction output columns: {out.columns.tolist()}")

    result = out[["timestamp", "target"]].rename(columns={"target": "predicted_heat"}).copy()
    for quantile in quantiles:
        source = find_column_by_numeric_value(out.columns, quantile)
        label = quantile_column_name(quantile)
        if source is not None:
            result[label] = out[source]
        elif math.isclose(quantile, 0.5, rel_tol=0.0, abs_tol=1e-12):
            result[label] = result["predicted_heat"]
        else:
            result[label] = np.nan
    if "q50" in result.columns:
        result["predicted_heat"] = result["q50"]
    return result


def quantile_column_name(value: float) -> str:
    percent = value * 100
    text = f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "_")
    if percent < 10 and not text.startswith("0"):
        text = f"0{text}"
    return f"q{text}"


def add_error_columns(df: pd.DataFrame, prediction_column: str = "predicted_heat") -> pd.DataFrame:
    out = df.copy()
    out["error"] = out[prediction_column] - out["actual_heat"]
    out["absolute_error"] = out["error"].abs()
    denominator = out["actual_heat"].abs() + out[prediction_column].abs()
    out["sape"] = np.where(denominator > 0, 2 * out["absolute_error"] / denominator, np.nan)
    return out


def metric_values(actual: pd.Series, prediction: pd.Series) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    prediction_values = prediction.to_numpy(dtype=float)
    error = prediction_values - actual_values
    absolute_error = np.abs(error)
    rmse = math.sqrt(float(np.mean(error**2)))
    actual_mean = float(np.mean(actual_values))
    total_sum_of_squares = float(np.sum((actual_values - actual_mean) ** 2))
    residual_sum_of_squares = float(np.sum(error**2))
    denominator = np.abs(actual_values) + np.abs(prediction_values)
    sape = np.where(denominator > 0, 2 * absolute_error / denominator, np.nan)
    return {
        "MAE": float(np.mean(absolute_error)),
        "RMSE": rmse,
        "R2": float(1 - residual_sum_of_squares / total_sum_of_squares) if total_sum_of_squares > 0 else np.nan,
        "CVRMSE_percent": float(rmse / actual_mean * 100) if actual_mean != 0 else np.nan,
        "sMAPE_percent": float(np.nanmean(sape) * 100),
        "mean_error": float(np.mean(error)),
        "median_absolute_error": float(np.median(absolute_error)),
        "max_absolute_error": float(np.max(absolute_error)),
    }


def calculate_summary(
    raw: pd.DataFrame,
    model_column: str = "model",
    prediction_column: str = "predicted_heat",
) -> pd.DataFrame:
    rows = []
    for model, model_group in raw.groupby(model_column, sort=True):
        row: dict[str, object] = {
            "model": model,
            "metric_scope": "all",
            "n_forecast_starts": int(model_group["forecast_start"].nunique()),
            "n_rows": int(len(model_group)),
        }
        row.update(metric_values(model_group["actual_heat"], model_group[prediction_column]))
        if "prediction_seconds" in model_group.columns:
            per_start = model_group.groupby("forecast_start", sort=False)["prediction_seconds"].first()
            row["prediction_seconds_total"] = float(per_start.sum())
            row["prediction_seconds_mean_per_forecast_start"] = float(per_start.mean())
        rows.append(row)
    return pd.DataFrame(rows)
