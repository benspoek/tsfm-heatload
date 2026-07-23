from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from utils import make_run_id, parse_weather_columns, write_metadata  # noqa: E402


class ExperimentUtilsTests(unittest.TestCase):
    def test_parse_weather_columns_strips_and_rejects_empty_values(self):
        self.assertEqual(parse_weather_columns(" temperature, solar ,"), ["temperature", "solar"])
        with self.assertRaisesRegex(ValueError, "at least one column"):
            parse_weather_columns(" , ")

    def test_explicit_run_id_is_sanitized_without_a_timestamp(self):
        self.assertEqual(make_run_id("publication run", "ignored"), "publication_run")

    def test_write_metadata_serializes_non_json_values(self):
        path = Mock()
        write_metadata(path, {"path": Path("input.csv")})
        serialized = path.write_text.call_args.args[0]
        self.assertEqual(json.loads(serialized), {"path": "input.csv"})


if __name__ == "__main__":
    unittest.main()
