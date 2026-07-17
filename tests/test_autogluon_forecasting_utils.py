from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from autogluon_forecasting_utils import (  # noqa: E402
    align_prediction_with_actual,
    build_autogluon_prediction_inputs,
)
from full_year_forecasting_utils import (  # noqa: E402
    HOURLY_STEP,
    TIMEZONE,
    add_synthetic_model_clock,
)


class AutoGluonInputContractTests(unittest.TestCase):
    def setUp(self) -> None:
        timestamps = pd.date_range(
            "2024-03-30 20:00:00",
            periods=12,
            freq=HOURLY_STEP,
            tz=TIMEZONE,
        )
        source = pd.DataFrame(
            {
                "timestamp": timestamps,
                "heat": range(100, 112),
                "temperature": range(12),
            }
        )
        self.data = add_synthetic_model_clock(source, HOURLY_STEP)
        self.forecast_start = self.data.loc[5, "timestamp"]
        self.history_start = self.data.loc[0, "timestamp"]

    def build_inputs(self, data: pd.DataFrame | None = None):
        return build_autogluon_prediction_inputs(
            data=self.data if data is None else data,
            forecast_start=self.forecast_start,
            prediction_steps=4,
            history_start=self.history_start,
            evaluation_year=2024,
            weather_columns=["temperature"],
            step=HOURLY_STEP,
        )

    def test_uses_regular_model_clock_and_preserves_real_timestamps(self) -> None:
        history, future_covariates, actual = self.build_inputs()
        self.assertNotIn("target", future_covariates.columns)
        self.assertEqual(history["timestamp"].iloc[-1] + HOURLY_STEP, future_covariates["timestamp"].iloc[0])
        self.assertTrue(
            pd.DatetimeIndex(future_covariates["timestamp"]).equals(
                pd.DatetimeIndex(actual["model_timestamp"])
            )
        )
        self.assertEqual(str(actual["timestamp"].dt.tz), TIMEZONE)
        self.assertListEqual(
            actual["timestamp"].tolist(),
            self.data.loc[5:8, "timestamp"].tolist(),
        )

    def test_rejects_target_history_that_does_not_reach_forecast_start(self) -> None:
        missing_last_history_row = self.data.drop(index=4).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "Target history.*ends at"):
            self.build_inputs(missing_last_history_row)

    def test_rejects_missing_future_covariate_timestamp(self) -> None:
        missing_future_row = self.data.drop(index=6).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "Real forecast horizon is not exactly aligned"):
            self.build_inputs(missing_future_row)

    def test_rejects_irregular_synthetic_forecast_clock(self) -> None:
        broken_clock = self.data.copy()
        broken_clock.loc[6, "model_timestamp"] += pd.Timedelta(minutes=30)
        with self.assertRaisesRegex(ValueError, "Synthetic forecast horizon is not exactly aligned"):
            self.build_inputs(broken_clock)

    def test_rejects_prediction_horizon_that_does_not_match_actual(self) -> None:
        _, _, actual = self.build_inputs()
        predicted = pd.DataFrame(
            {
                "model_timestamp": actual["model_timestamp"].iloc[:-1],
                "predicted_heat": [1.0, 2.0, 3.0],
                "model": "test-model",
            }
        )
        with self.assertRaisesRegex(ValueError, "prediction horizon is not exactly aligned"):
            align_prediction_with_actual(predicted, actual, self.forecast_start)


if __name__ == "__main__":
    unittest.main()
