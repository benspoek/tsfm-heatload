from __future__ import annotations

import json
import platform
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from full_year_forecasting_utils import git_metadata, safe_name


AUTOGLUON_PACKAGES = (
    "autogluon",
    "autogluon.timeseries",
    "autogluon.tabular",
    "lightgbm",
    "xgboost",
    "pandas",
    "numpy",
)
CHRONOS_PACKAGES = (
    "chronos-forecasting",
    "torch",
    "transformers",
    "huggingface_hub",
    "pandas",
    "numpy",
)
TABPFN_PACKAGES = (
    "tabpfn-time-series",
    "tabpfn",
    "pandas",
    "numpy",
    "scikit-learn",
)


def parse_weather_columns(value: str) -> list[str]:
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise ValueError("--weather-columns must contain at least one column.")
    return columns


def make_run_id(explicit_run_id: str | None, run_name: str) -> str:
    if explicit_run_id:
        return safe_name(explicit_run_id)
    created = datetime.now().strftime("%Y%m%d_%H%M%S")
    return safe_name(f"{created}_{run_name}")


def package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def metadata_envelope(
    *,
    run_id: str,
    run_name: str | None,
    run_dir: Path,
    script_path: str | Path,
    packages: Iterable[str],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(script_path).name,
        "command": sys.argv,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": git_metadata(),
        "package_versions": package_versions(packages),
        "run_dir": str(run_dir),
    }


def write_metadata(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, default=str) + "\n", encoding="utf-8")
