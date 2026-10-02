"""Protect run timing, location-specific caching and source-value decoding."""
import argparse
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_flensburg_ecmwf_coherent_predictions as builder


class FlensburgPredictionPreparationTests(unittest.TestCase):
    def test_run_assignment_keeps_eleven_hour_lag_across_dst(self):
        for issue_text in (
            "2024-03-30T00:00:00+01:00",
            "2024-04-01T01:00:00+02:00",
            "2024-10-28T00:00:00+01:00",
        ):
            issue = pd.Timestamp(issue_text)
            run = builder.assigned_run_initialization(issue)
            self.assertEqual(run.hour, 12)
            self.assertEqual(issue.tz_convert("UTC") - run, pd.Timedelta(hours=11))
            self.assertEqual(run.date(), issue.tz_convert("UTC").date())
        with self.assertRaisesRegex(ValueError, "six hours"):
            builder.assigned_run_initialization(pd.Timestamp("2024-04-01T15:00:00Z"))

    def test_cache_cannot_reuse_a_munich_response_for_flensburg(self):
        run = pd.Timestamp("2024-06-06T12:00:00Z")
        flensburg = builder.request_url(argparse.Namespace(latitude=54.7937, longitude=9.4469), run)
        munich = builder.request_url(argparse.Namespace(latitude=48.127, longitude=11.604), run)
        cache_dir = Path("unused_cache")
        self.assertNotEqual(builder.cache_path(cache_dir, run, flensburg),
                            builder.cache_path(cache_dir, run, munich))
        self.assertEqual(builder.cache_path(cache_dir, run, flensburg),
                         builder.cache_path(cache_dir, run, flensburg))

    def test_temperature_values_are_keyed_to_utc_valid_time(self):
        payload = {
            "hourly_units": {"temperature_2m": "\u00b0C"},
            "hourly": {
                "time": ["2024-06-06T23:00", "2024-06-07T00:00"],
                "temperature_2m": [11.2, 10.4],
            },
        }
        values = builder.payload_temperature(payload)
        self.assertEqual(values.loc[pd.Timestamp("2024-06-06T23:00:00Z")], 11.2)
        self.assertEqual(values.loc[pd.Timestamp("2024-06-07T00:00:00Z")], 10.4)

    def test_duplicate_valid_times_and_wrong_units_are_rejected(self):
        payload = {
            "hourly_units": {"temperature_2m": "\u00b0C"},
            "hourly": {
                "time": ["2024-06-06T23:00", "2024-06-06T23:00"],
                "temperature_2m": [11.2, 10.4],
            },
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            builder.payload_temperature(payload)
        payload["hourly_units"]["temperature_2m"] = "\u00b0F"
        with self.assertRaisesRegex(ValueError, "unit"):
            builder.payload_temperature(payload)


if __name__ == "__main__":
    unittest.main()
