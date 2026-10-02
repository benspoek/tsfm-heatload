from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ecmwf_weather_sensitivity as weather
from full_year_forecasting_utils import add_synthetic_model_clock, HOURLY_STEP


def synthetic_data(start="2024-06-06 01:00", periods=48):
    timestamps = pd.date_range(start, periods=periods, freq="h", tz="Europe/Berlin")
    issue_anchor = pd.Timestamp("2024-01-01", tz="Europe/Berlin")
    day = pd.Timedelta(hours=24)
    issues = pd.DatetimeIndex([issue_anchor + int((timestamp - issue_anchor) // day) * day for timestamp in timestamps])
    runs = issues.tz_convert("UTC").normalize() + pd.Timedelta(hours=12)
    data = pd.DataFrame({
        "timestamp": timestamps,
        "heat": 100.0 + np.arange(periods),
        "temperature_observed": 1.0 + np.arange(periods),
        "temperature_predicted": 1001.0 + np.arange(periods),
        "forecast_issue_time_local": issues,
        "forecast_run_initialization_utc": runs,
        "forecast_lead_time_hours": (timestamps.tz_convert("UTC") - runs) / HOURLY_STEP,
    })
    before_archive = data["timestamp"].lt(pd.Timestamp("2024-03-15", tz="Europe/Berlin"))
    data.loc[before_archive, ["temperature_predicted", "forecast_lead_time_hours"]] = np.nan
    data.loc[before_archive, "forecast_issue_time_local"] = pd.NaT
    data.loc[before_archive, "forecast_run_initialization_utc"] = pd.NaT
    return add_synthetic_model_clock(data, HOURLY_STEP)


class CoherentInputTests(unittest.TestCase):
    def test_mixed_runs_cannot_form_one_future_vector(self):
        data = synthetic_data()
        data.loc[30, "forecast_run_initialization_utc"] -= pd.Timedelta(hours=12)
        with self.assertRaisesRegex(ValueError, "mixes ECMWF"):
            weather.validate_coherent_weather(data)

    def test_initialization_after_issue_is_rejected(self):
        data = synthetic_data()
        data.loc[24:, "forecast_run_initialization_utc"] += pd.Timedelta(hours=12)
        with self.assertRaisesRegex(ValueError, "designated 12 UTC"):
            weather.validate_coherent_weather(data)

    def test_fixed_lead_labels_are_rejected_for_coherent_trajectory(self):
        data = synthetic_data()
        data["forecast_lead_time_hours"] = 24.0
        with self.assertRaisesRegex(ValueError, "inconsistent forecast lead"):
            weather.validate_coherent_weather(data)

    def test_wrong_issue_and_incomplete_trajectory_are_rejected(self):
        for change in ("wrong_issue", "missing_row"):
            data = synthetic_data()
            if change == "wrong_issue":
                data.loc[25, "forecast_issue_time_local"] -= pd.Timedelta(hours=24)
            else:
                data = data.drop(index=25)
            with self.subTest(change=change), self.assertRaises(ValueError):
                weather.validate_coherent_weather(data)

    def test_history_must_be_assigned_to_an_earlier_issue(self):
        data = synthetic_data()
        issue = data.timestamp.iloc[24]
        data.loc[0, "forecast_issue_time_local"] = issue
        with self.assertRaisesRegex(ValueError, "preceding forecast issues"):
            weather.build_window(data, issue, 24, "forecast_only")

    def test_hash_accepts_windows_checkout_but_rejects_changed_data(self):
        original = b"date,temperature_predicted\n2024-06-07,11.0\n"
        metadata = {"output_sha256": hashlib.sha256(original).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.csv"
            for content in (original, original.replace(b"\n", b"\r\n")):
                path.write_bytes(content)
                weather.verify_weather_hash(path, metadata)
            path.write_bytes(original.replace(b"11.0", b"12.0"))
            with self.assertRaisesRegex(ValueError, "does not match"):
                weather.verify_weather_hash(path, metadata)

    def test_dst_uses_elapsed_hours_and_unique_model_clock(self):
        for start in ("2024-03-30 00:00", "2024-10-26 01:00"):
            data = synthetic_data(start=start, periods=72)
            weather.validate_coherent_weather(data)
            issue = data.timestamp.iloc[24]
            context, future, _ = weather.build_window(data, issue, 24, "forecast_only")
            self.assertEqual(len(future), 24)
            self.assertFalse(future.timestamp.duplicated().any())
            self.assertEqual(future.timestamp.iloc[0] - context.timestamp.iloc[-1], HOURLY_STEP)
            self.assertTrue(data.timestamp.diff().dropna().eq(HOURLY_STEP).all())
        self.assertEqual(synthetic_data("2024-03-30 00:00", 72).timestamp.iloc[48].hour, 1)
        self.assertEqual(synthetic_data("2024-10-26 01:00", 72).timestamp.iloc[48].hour, 0)


class CovariateSafetyTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()
        self.issue = self.data.timestamp.iloc[24]

    def test_future_heat_never_changes_model_inputs_in_any_mode(self):
        changed = self.data.copy()
        changed.loc[24:, "heat"] = 999999.0
        for mode in weather.MODES:
            with self.subTest(mode=mode):
                context, future, _ = weather.build_window(self.data, self.issue, 24, mode)
                context2, future2, _ = weather.build_window(changed, self.issue, 24, mode)
                pd.testing.assert_frame_equal(context, context2)
                pd.testing.assert_frame_equal(future, future2)
                self.assertNotIn("target", future)
                self.assertNotIn("heat", future)

    def test_future_measurements_cannot_change_predicted_weather_inputs(self):
        changed = self.data.copy()
        changed.loc[24:, "temperature_observed"] = 999999.0
        for mode in weather.MODES[1:]:
            with self.subTest(mode=mode):
                context, future, _ = weather.build_window(self.data, self.issue, 24, mode)
                context2, future2, _ = weather.build_window(changed, self.issue, 24, mode)
                pd.testing.assert_frame_equal(context, context2)
                pd.testing.assert_frame_equal(future, future2)

    def test_selected_dual_modes_preserve_distinct_past_sources(self):
        context, future, _ = weather.build_window(self.data, self.issue, 24, "dual_forecast_fill")
        np.testing.assert_array_equal(context.temperature_observed, self.data.temperature_observed.iloc[:24])
        np.testing.assert_array_equal(context.temperature_predicted, self.data.temperature_predicted.iloc[:24])
        np.testing.assert_array_equal(future.temperature_observed, self.data.temperature_predicted.iloc[24:])
        np.testing.assert_array_equal(future.temperature_predicted, self.data.temperature_predicted.iloc[24:])
        context2, future2, _ = weather.build_window(self.data, self.issue, 24, "dual_history")
        pd.testing.assert_frame_equal(context, context2)
        self.assertNotIn("temperature_observed", future2)

    def test_measured_reference_intentionally_uses_future_measurements(self):
        _, future, _ = weather.build_window(self.data, self.issue, 24, "measured_only")
        np.testing.assert_array_equal(future.temperature, self.data.temperature_observed.iloc[24:])


class MatchedMaskTests(unittest.TestCase):
    def test_reference_and_predicted_modes_share_complete_12_week_mask(self):
        data = synthetic_data("2023-10-09 01:00", 10800)
        candidates, matched = weather.matched_starts(data, 2024, 2016)
        self.assertEqual(len(candidates), 366)
        self.assertEqual(len(matched), 208)
        self.assertEqual(matched.forecast_start.iloc[0], pd.Timestamp("2024-06-07 01:00", tz="Europe/Berlin"))
        self.assertEqual(matched.forecast_start.iloc[-1], pd.Timestamp("2024-12-31 00:00", tz="Europe/Berlin"))
        earlier = pd.Timestamp("2024-06-06 01:00", tz="Europe/Berlin")
        for mode in weather.MODES:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "Incomplete matched"):
                    weather.build_window(data, earlier, 2016, mode)
                context, future, _ = weather.build_window(data, matched.forecast_start.iloc[0], 2016, mode)
                self.assertEqual(len(context), 2016)
                self.assertEqual(len(future), 24)

    def test_model_defaults_and_context_cap_preserve_scientific_comparison(self):
        self.assertEqual(weather.parse_args("tabpfn", []).covariate_modes, ["measured_only", "dual_forecast_fill"])
        self.assertEqual(weather.parse_args("chronos2", []).covariate_modes, ["measured_only", "dual_history"])
        with contextlib.redirect_stderr(io.StringIO()):
            for args, model in ((["--max-context-steps", "1024"], "chronos2"), (["--covariate-modes", "dual_history"], "tabpfn"), (["--forecast-every-hours", "12"], "tabpfn")):
                with self.subTest(model=model, args=args), self.assertRaises(SystemExit):
                    weather.parse_args(model, args)
        self.assertNotIn("dual_history", weather.parse_args("tabpfn", ["--covariate-modes", "all"]).covariate_modes)


class ModelAdapterTests(unittest.TestCase):
    def test_both_backbones_receive_selected_features_and_no_future_targets(self):
        data = synthetic_data()
        issue = data.timestamp.iloc[24]
        for model in ("tabpfn", "chronos2"):
            class RecordingPipeline:
                max_context_length = 24

                def predict_df(self, *args, **kwargs):
                    self.context = kwargs["context_df"] if "context_df" in kwargs else args[0]
                    self.future = kwargs["future_df"]
                    self.kwargs = kwargs
                    return pd.DataFrame({"timestamp": self.future.timestamp.to_numpy(), "target": np.arange(24) + 124.0})

            pipeline = RecordingPipeline()
            mode = weather.SELECTED_MODE[model]
            raw, metrics = weather.predict_one(pipeline, model, data, issue, 24, mode)
            with self.subTest(model=model):
                self.assertNotIn("target", pipeline.future)
                self.assertEqual(len(raw), 24)
                self.assertEqual(metrics["MAE"], 0.0)
                self.assertTrue(raw.forecast_start.eq(issue.isoformat()).all())
                self.assertEqual(len(pipeline.context), 24)
                if model == "tabpfn":
                    self.assertIn("quantiles", pipeline.kwargs)
                    self.assertEqual(list(pipeline.future.columns), ["item_id", "timestamp", "temperature_observed", "temperature_predicted"])
                else:
                    self.assertEqual(pipeline.kwargs["prediction_length"], 24)
                    self.assertEqual(pipeline.kwargs["target"], "target")
                    self.assertEqual(list(pipeline.future.columns), ["item_id", "timestamp", "temperature_predicted"])


if __name__ == "__main__":
    unittest.main()
