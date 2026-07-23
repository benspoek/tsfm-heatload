from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from full_year_forecasting_utils import read_weather_csv


DEFAULT_WEATHER_PATH = Path("flensburg/weather/flensburg_weather_temperature.csv")
DEFAULT_OUTPUT_PATH = Path("flensburg/weather/representative_weeks_2024.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the hottest, coldest, and largest-temperature-range complete "
            "Monday-to-Sunday ISO weeks from the Flensburg weather data."
        )
    )
    parser.add_argument("--weather-path", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--year", type=int, default=2024)
    return parser.parse_args()


def load_weather(path: Path) -> pd.DataFrame:
    return read_weather_csv(path, ["temperature"])


def add_iso_week_columns(weather: pd.DataFrame) -> pd.DataFrame:
    out = weather.copy()
    iso = out["timestamp"].dt.isocalendar()
    out["iso_year"] = iso["year"].astype(int)
    out["iso_week"] = iso["week"].astype(int)
    out["iso_weekday"] = iso["day"].astype(int)
    out["week_id"] = out["iso_year"].astype(str) + "-W" + out["iso_week"].astype(str).str.zfill(2)
    return out


def is_complete_local_week(week: pd.DataFrame) -> bool:
    timestamps = pd.DatetimeIndex(week["timestamp"].sort_values())
    if timestamps.empty:
        return False

    week_start = timestamps[0]
    starts_monday_midnight = (
        week_start.isoweekday() == 1
        and week_start.hour == 0
        and week_start.minute == 0
        and week_start.second == 0
    )
    if not starts_monday_midnight:
        return False

    next_monday = week_start + pd.DateOffset(days=7)
    expected = pd.date_range(week_start, next_monday, freq="1h", inclusive="left")
    return timestamps.equals(expected)


def build_week_features(weather: pd.DataFrame, year: int) -> pd.DataFrame:
    weather = add_iso_week_columns(weather)
    weather = weather[weather["iso_year"] == year]
    rows: list[dict[str, object]] = []

    for week_id, week in weather.groupby("week_id", sort=True):
        week = week.sort_values("timestamp")
        if not is_complete_local_week(week):
            continue

        temperature = week["temperature"]
        week_start = week["timestamp"].iloc[0]
        week_end = week["timestamp"].iloc[-1]
        rows.append(
            {
                "week_id": str(week_id),
                "iso_year": int(week["iso_year"].iloc[0]),
                "iso_week": int(week["iso_week"].iloc[0]),
                "week_start": week_start.tz_localize(None),
                "week_end": week_end.tz_localize(None),
                "n_rows": int(len(week)),
                "temperature_mean": float(temperature.mean()),
                "temperature_std": float(temperature.std(ddof=0)),
                "temperature_min": float(temperature.min()),
                "temperature_max": float(temperature.max()),
                "temperature_q10": float(temperature.quantile(0.1)),
                "temperature_q50": float(temperature.quantile(0.5)),
                "temperature_q90": float(temperature.quantile(0.9)),
                "temperature_range": float(temperature.max() - temperature.min()),
            }
        )

    if not rows:
        raise ValueError(f"No complete Monday-to-Sunday ISO weeks found for {year}.")
    return pd.DataFrame(rows)


def select_representative_weeks(features: pd.DataFrame) -> pd.DataFrame:
    selections = (
        ("hottest", "temperature_mean", "max"),
        ("coldest", "temperature_mean", "min"),
        ("highest_temperature_fluctuation", "temperature_range", "max"),
    )
    rows = []
    for label, column, direction in selections:
        index = features[column].idxmax() if direction == "max" else features[column].idxmin()
        row = features.loc[index].copy()
        row["selection"] = label
        rows.append(row)
    selected = pd.DataFrame(rows).reset_index(drop=True)
    return selected[["selection", *features.columns]]


def main() -> None:
    args = parse_args()
    weather = load_weather(args.weather_path)
    features = build_week_features(weather, args.year)
    selected = select_representative_weeks(features)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_path, index=False, float_format="%.6f")

    print(f"Complete ISO weeks checked: {len(features)}")
    print(f"Saved: {args.output_path}")
    print(
        selected[
            [
                "selection",
                "week_id",
                "week_start",
                "week_end",
                "n_rows",
                "temperature_mean",
                "temperature_range",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
