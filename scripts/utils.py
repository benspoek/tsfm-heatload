from __future__ import annotations

import json
import os
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
    "wandb",
)
TABPFN_PACKAGES = (
    "tabpfn-time-series",
    "tabpfn",
    "pandas",
    "numpy",
    "scikit-learn",
    "wandb",
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


def initialize_wandb(
    args: Any,
    run_id: str,
    config: Mapping[str, Any],
    tags: Iterable[str] = (),
):
    if args.disable_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install it with: python -m pip install -r requirements.txt") from exc

    configured_tags = [tag.strip() for tag in tags if tag.strip()]
    environment_tags = [
        tag.strip() for tag in os.environ.get("WANDB_RUN_TAGS", "").split(",") if tag.strip()
    ]
    all_tags = sorted(set([*configured_tags, *environment_tags]))
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=getattr(args, "wandb_run_name", None) or getattr(args, "run_name", None) or run_id,
        group=getattr(args, "wandb_group", None) or os.environ.get("WANDB_RUN_GROUP") or None,
        tags=all_tags or None,
        config=dict(config),
    )


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
