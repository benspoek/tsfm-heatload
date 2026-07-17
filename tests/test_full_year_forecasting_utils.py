from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from full_year_forecasting_utils import (  # noqa: E402
    HOURLY_STEP,
    QUARTER_STEP,
    TIMEZONE,
    UnsupportedDataCadenceError,
    aggregate_to_step,
    align_timeseries_frames,
    forecast_starts_for_selected_weeks,
    load_residual_multiresolution_data,
    load_selected_weeks,
    validate_complete_year,
    validate_timeseries_frame,
)


def frame(start: str, periods: int, freq: str, column: str = "value") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=periods, freq=freq, tz=TIMEZONE),
            column: range(periods),
        }
    )


class AggregationTests(unittest.TestCase):
    def test_hourly_input_stays_hourly(self) -> None:
        hourly = frame("2024-01-01", periods=4, freq="1h")
        result = aggregate_to_step(hourly, ["value"], HOURLY_STEP, label="hourly test")
        self.assertEqual(len(result), 4)
        self.assertListEqual(result["value"].tolist(), [0.0, 1.0, 2.0, 3.0])

    def test_quarter_hour_input_is_aggregated_by_inferred_cadence(self) -> None:
        quarter = frame("2024-01-01", periods=8, freq="15min")
        result = aggregate_to_step(quarter, ["value"], HOURLY_STEP, label="quarter test")
        self.assertEqual(len(result), 2)
        self.assertListEqual(result["value"].tolist(), [1.5, 5.5])

    def test_duplicate_and_missing_timestamps_are_rejected(self) -> None:
        duplicated = frame("2024-01-01", periods=4, freq="1h")
        duplicated.loc[2, "timestamp"] = duplicated.loc[1, "timestamp"]
        with self.assertRaisesRegex(ValueError, "duplicate timestamps"):
            validate_timeseries_frame(duplicated, ["value"], "duplicate test")

        missing = frame("2024-01-01", periods=4, freq="1h").drop(index=2).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "not regular"):
            validate_timeseries_frame(missing, ["value"], "missing-row test")

    def test_alignment_rejects_an_internal_gap(self) -> None:
        left = frame("2024-01-01", periods=4, freq="1h", column="left")
        right = frame("2024-01-01", periods=4, freq="1h", column="right")
        right = right.drop(index=2).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "not regular"):
            align_timeseries_frames(
                left,
                right,
                left_value_columns=["left"],
                right_value_columns=["right"],
                step=HOURLY_STEP,
                label="alignment test",
            )


class ResidualCadenceContractTests(unittest.TestCase):
    def test_hourly_heat_is_rejected_before_weather_is_read(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedDataCadenceError,
            r"Unsupported cadence for 15-minute residual forecasting:.*is hourly.*native 15-minute",
        ):
            load_residual_multiresolution_data(
                FIXTURE_ROOT / "hourly_heat.csv",
                FIXTURE_ROOT / "missing_weather.csv",
                ["temperature"],
            )

    def test_quarter_hour_heat_and_weather_are_supported(self) -> None:
        quarter_data, hourly_data = load_residual_multiresolution_data(
            FIXTURE_ROOT / "quarter_heat.csv",
            FIXTURE_ROOT / "quarter_weather.csv",
            ["temperature"],
        )

        self.assertEqual(len(quarter_data), 8)
        self.assertEqual(len(hourly_data), 2)


class CalendarValidationTests(unittest.TestCase):
    def test_complete_local_year_handles_dst(self) -> None:
        timestamps = pd.date_range(
            pd.Timestamp("2024-01-01", tz=TIMEZONE),
            pd.Timestamp("2025-01-01", tz=TIMEZONE),
            freq=HOURLY_STEP,
            inclusive="left",
        )
        data = pd.DataFrame({"timestamp": timestamps, "heat": 1.0})
        validated = validate_complete_year(data, 2024, HOURLY_STEP, ["heat"], "DST test")
        self.assertEqual(len(validated), 8784)
        self.assertEqual(
            len(validated[validated["timestamp"].dt.strftime("%Y-%m-%d").eq("2024-10-27")]),
            25,
        )

    def test_complete_year_rejects_one_missing_row(self) -> None:
        timestamps = pd.date_range(
            pd.Timestamp("2024-01-01", tz=TIMEZONE),
            pd.Timestamp("2025-01-01", tz=TIMEZONE),
            freq=QUARTER_STEP,
            inclusive="left",
        ).delete(500)
        data = pd.DataFrame({"timestamp": timestamps, "heat": 1.0})
        with self.assertRaisesRegex(ValueError, "does not contain a complete 2024"):
            validate_complete_year(data, 2024, QUARTER_STEP, ["heat"], "missing-year test")


class FlensburgWeekTests(unittest.TestCase):
    def test_checked_in_weeks_are_monday_to_sunday_and_generate_21_daily_starts(self) -> None:
        selected_path = REPO_ROOT / "flensburg" / "weather" / "representative_weeks_2024.csv"
        selected = load_selected_weeks(selected_path)
        self.assertSetEqual(
            set(selected["selection"]),
            {"hottest", "coldest", "highest_temperature_fluctuation"},
        )
        self.assertTrue(all(start.isoweekday() == 1 for start in selected["week_start"]))

        starts = forecast_starts_for_selected_weeks(selected, prediction_hours=24)
        self.assertEqual(len(starts), 21)
        for selection, week_id, forecast_start in starts:
            week_start = selected.loc[selected["selection"].eq(selection), "week_start"].iloc[0]
            self.assertEqual(week_id, f"{week_start.isocalendar().year}-W{week_start.isocalendar().week:02d}")
            self.assertGreaterEqual(forecast_start, week_start)
            self.assertLessEqual(forecast_start + pd.Timedelta(hours=24), week_start + pd.DateOffset(days=7))


if __name__ == "__main__":
    unittest.main()
