"""Prepare the public Flensburg IFS hindcast input without running heat forecasts."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from full_year_forecasting_utils import (
    HOURLY_STEP,
    TIMEZONE,
    forecast_starts_from_rows,
    load_merged_data,
    sha256_file,
    validate_regular_rows,
)

API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs"
ARCHIVE_FIRST_RUN = pd.Timestamp("2024-03-14T00:00:00Z")
DEFAULT_OUTPUT = Path("flensburg/weather/flensburg_temperature_observed_vs_ecmwf_coherent.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heat-path", type=Path, default=Path("flensburg/demand/heat/heat_dh.csv"))
    parser.add_argument("--observed-weather-path", type=Path,
                        default=Path("flensburg/weather/flensburg_weather_temperature.csv"))
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-path", type=Path,
                        default=DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_metadata.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/flensburg_ifs_single_runs_cache"))
    parser.add_argument("--latitude", type=float, default=54.7937)
    parser.add_argument("--longitude", type=float, default=9.4469)
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-request-attempts", type=int, default=4)
    parser.add_argument("--request-delay-seconds", type=float, default=0.2)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def assigned_run_initialization(issue_time: pd.Timestamp) -> pd.Timestamp:
    """Select the preceding 12 UTC run on the heat issuance's UTC calendar date."""
    if issue_time.tzinfo is None:
        raise ValueError("Heat issuance must include a timezone.")
    issue_utc = issue_time.tz_convert("UTC")
    run = issue_utc.normalize() + pd.Timedelta(hours=12)
    if (issue_utc - run) < pd.Timedelta(hours=6):
        raise ValueError(f"Run {run} does not precede issuance {issue_time} by at least six hours.")
    return run


def request_url(args: argparse.Namespace, run: pd.Timestamp) -> str:
    params = {
        "latitude": args.latitude,
        "longitude": args.longitude,
        "hourly": "temperature_2m",
        "models": MODEL,
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "forecast_hours": 48,
        "timezone": "UTC",
        "temperature_unit": "celsius",
    }
    return API_URL + "?" + urllib.parse.urlencode(params)


def cache_path(cache_dir: Path, run: pd.Timestamp, url: str) -> Path:
    # The URL digest includes coordinates, model and all query parameters. A Munich
    # cache entry therefore cannot be reused for Flensburg, even in a shared folder.
    request_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{MODEL}_{run.strftime('%Y%m%dT%H%MZ')}_{request_key}.json"


def download_payload(args: argparse.Namespace, run: pd.Timestamp) -> tuple[dict, str, str]:
    url = request_url(args, run)
    path = cache_path(args.cache_dir, run, url)
    if path.exists() and not args.overwrite_cache:
        raw = path.read_bytes()
        return json.loads(raw), url, hashlib.sha256(raw).hexdigest()

    last_error: Exception | None = None
    for attempt in range(args.max_request_attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "tsfm-heatload-reproduction/1.0.1"})
            with urllib.request.urlopen(request, timeout=args.request_timeout_seconds) as response:
                raw = response.read()
            payload = json.loads(raw)
            if not isinstance(payload.get("hourly"), dict):
                raise ValueError(f"API response lacks hourly data for {run}.")
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            time.sleep(args.request_delay_seconds)
            return payload, url, hashlib.sha256(raw).hexdigest()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < args.max_request_attempts:
                delay = min(2 ** (attempt + 1), 30)
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                    retry_after = exc.headers.get("Retry-After", "")
                    if retry_after.isdigit():
                        delay = min(max(delay, int(retry_after)), 60)
                print(f"Retry {attempt + 1}/{args.max_request_attempts - 1} for {run}: {exc}", flush=True)
                time.sleep(delay)
    raise RuntimeError(f"Download failed for {run}: {last_error}")


def payload_temperature(payload: dict) -> pd.Series:
    hourly = payload.get("hourly", {})
    if "time" not in hourly or "temperature_2m" not in hourly:
        raise ValueError("API payload lacks hourly time or temperature_2m.")
    units = payload.get("hourly_units", {})
    if units.get("temperature_2m") not in ("\u00b0C", "C", "celsius"):
        raise ValueError(f"Unexpected temperature unit: {units.get('temperature_2m')}")
    values = pd.Series(
        pd.to_numeric(pd.Series(hourly["temperature_2m"]), errors="coerce").to_numpy(),
        index=pd.to_datetime(hourly["time"], utc=True, errors="raise"),
    )
    if values.index.duplicated().any():
        raise ValueError("API payload contains duplicate valid timestamps.")
    return values


def prepare_base(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_merged_data(args.heat_path, args.observed_weather_path, ["temperature"], HOURLY_STEP)
    validate_regular_rows(data, HOURLY_STEP, "Flensburg hourly heat and measured temperature")
    candidates = forecast_starts_from_rows(data, 2024, 24, 24)
    if len(candidates) != 366:
        raise ValueError(f"Expected 366 complete 2024 Flensburg forecast starts; found {len(candidates)}.")
    first_issue = candidates["forecast_start"].iloc[0]
    last_end = candidates["horizon_end"].iloc[-1]
    base = data.loc[
        data["timestamp"].ge(first_issue - pd.Timedelta(hours=2016))
        & data["timestamp"].le(last_end),
        ["timestamp", "temperature"],
    ].rename(columns={"timestamp": "date", "temperature": "temperature_observed"}).reset_index(drop=True)
    if len(base) != 2016 + 366 * 24:
        raise ValueError("The full-year grid requires a complete additional 2016-hour observed context.")
    starts = candidates.loc[
        candidates["forecast_start"].map(assigned_run_initialization).ge(ARCHIVE_FIRST_RUN)
    ].reset_index(drop=True)
    if len(starts) != 292:
        raise ValueError(f"Expected 292 archive-covered trajectories; found {len(starts)}.")
    return base, candidates, starts


def build_forecast_rows(args: argparse.Namespace, starts: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    parts = []
    records = []
    for number, issue in enumerate(starts["forecast_start"], start=1):
        run = assigned_run_initialization(issue)
        lag = (issue.tz_convert("UTC") - run) / HOURLY_STEP
        if lag != 11:
            raise ValueError(f"Expected the 2024 23 UTC issue cadence; got lag {lag} for {issue}.")
        payload, url, payload_hash = download_payload(args, run)
        valid_local = pd.date_range(issue, periods=24, freq="h")
        valid_utc = valid_local.tz_convert("UTC")
        temperature = payload_temperature(payload).reindex(valid_utc)
        if not np.isfinite(temperature.to_numpy()).all():
            raise ValueError(f"Run {run} has missing or nonfinite temperatures in its required 24-hour horizon.")
        lead = (valid_utc - run) / HOURLY_STEP
        if not np.array_equal(lead, np.arange(11, 35)):
            raise ValueError(f"Unexpected IFS lead-time grid for {issue}.")
        parts.append(pd.DataFrame({
            "date": valid_local,
            "temperature_predicted": temperature.to_numpy(),
            "forecast_issue_time_local": issue,
            "forecast_run_initialization_utc": run,
            "forecast_lead_time_hours": lead,
        }))
        records.append({
            "forecast_issue_time_local": issue.isoformat(),
            "run_initialization_utc": run.isoformat(),
            "availability_lag_hours": float(lag),
            "api_url": url,
            "payload_sha256": payload_hash,
            "response_latitude": payload.get("latitude"),
            "response_longitude": payload.get("longitude"),
            "response_elevation_m": payload.get("elevation"),
            "response_timezone": payload.get("timezone"),
        })
        print(f"[{number}/{len(starts)}] issue={issue.isoformat()} run={run.isoformat()}", flush=True)
    forecasts = pd.concat(parts, ignore_index=True)
    if forecasts["date"].duplicated().any() or len(forecasts) != 7008:
        raise ValueError("Coherent daily trajectories must cover 7008 distinct valid times.")
    return forecasts, records


def main() -> None:
    args = parse_args()
    if args.max_request_attempts < 1 or args.request_delay_seconds < 0 or args.request_timeout_seconds <= 0:
        raise ValueError("Request attempts and timeout must be positive; delay must be nonnegative.")
    base, candidates, starts = prepare_base(args)
    matched_first = starts["forecast_start"].iloc[0] + pd.Timedelta(hours=2016)
    matched = candidates.loc[candidates["forecast_start"].ge(matched_first)]
    print(f"Full-year candidate starts: {len(candidates)}", flush=True)
    print(f"Archive-covered trajectories: {len(starts)} ({starts['forecast_start'].iloc[0]} to {starts['forecast_start'].iloc[-1]})", flush=True)
    print(f"Matched starts with 12-week predicted-temperature context: {len(matched)} ({matched_first} to {matched['forecast_start'].iloc[-1]})", flush=True)
    print(f"Observed rows including pre-2024 context: {len(base)}", flush=True)
    if args.dry_run:
        print("Dry run complete. No API requests or files written.")
        return

    predictions, requests = build_forecast_rows(args, starts)
    output = base.merge(predictions, on="date", how="left", validate="one_to_one")
    if output["temperature_predicted"].notna().sum() != 7008:
        raise ValueError("Not all downloaded trajectory timestamps occur in the observed base grid.")
    if len(matched) != 208:
        raise ValueError("Expected 208 matched evaluation starts.")
    for column in ("date", "forecast_issue_time_local", "forecast_run_initialization_utc"):
        output[column] = output[column].map(lambda value: value.isoformat() if pd.notna(value) else "")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    # Canonical LF bytes keep the archived hash stable across Git checkouts on
    # Windows and Linux. Readers may normalize checkout CRLF back to LF.
    output.to_csv(args.output_path, index=False, lineterminator="\n")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "output_path": args.output_path.as_posix(),
        "output_sha256": sha256_file(args.output_path),
        "inputs": {
            "heat_path": args.heat_path.as_posix(), "heat_sha256": sha256_file(args.heat_path),
            "observed_weather_path": args.observed_weather_path.as_posix(),
            "observed_weather_sha256": sha256_file(args.observed_weather_path),
        },
        "model_and_api": {
            "provider": "ECMWF", "model": "IFS HRES", "open_meteo_model_parameter": MODEL,
            "api": API_URL, "documentation": "https://open-meteo.com/en/docs/single-runs-api",
            "latitude": args.latitude, "longitude": args.longitude,
            "temperature_variable": "temperature_2m", "temperature_unit": "celsius",
            "api_timezone": "UTC", "spatial_selection": "Open-Meteo terrain-optimized point extraction",
            "archive_qualification": "The 2024 archive contains IFS Cycle 49R1 retrospective hindcasts, not verified weather forecasts delivered operationally in 2024. Run initialization times are not actual historical publication timestamps.",
        },
        "license": {
            "spdx": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/",
            "data_license_source": "https://open-meteo.com/en/licence",
            "attribution": "Weather data by Open-Meteo.com (https://open-meteo.com/); ECMWF IFS HRES hindcasts.",
            "processing": "Hourly temperature_2m values were selected from individual 12 UTC runs, restricted to each matching 24-hour heat forecast horizon and joined with independently supplied DWD measured temperatures. Model values were not modified.",
            "measured_temperature_license": "See flensburg/weather/flensburg_weather_temperature_metadata.json and THIRD_PARTY_NOTICES.md for DWD provenance and license.",
        },
        "forecast_protocol": {
            "timezone": TIMEZONE, "year": 2024, "context_hours": 2016, "prediction_hours": 24,
            "forecast_every_hours": 24, "run_cycle_hour_utc": 12,
            "heat_forecast_starts": "24 elapsed hours on the full 2024 hourly grid; 23:00 UTC throughout, corresponding to 00:00 Europe/Berlin in winter and 01:00 in summer.",
            "run_assignment": "12 UTC on the UTC calendar date of the 23 UTC heat-forecast issuance (the preceding local calendar day).",
            "minimum_assumed_availability_lag_hours": 6.0,
            "availability_lag_min_hours": 11.0, "availability_lag_max_hours": 11.0,
            "lead_time_min_hours": 11.0, "lead_time_max_hours": 34.0,
            "future_vector": "All 24 future temperatures for an issuance come from one ECMWF run, with leads of 11 through 34 hours.",
            "historical_predictions": "Each historical value retains the trajectory assigned to its corresponding daily heat-forecast issuance; no later model runs replace it.",
            "operational_availability_assumption": "Six-hour minimum computation/dissemination lag is assumed. The hindcast archive does not provide observed historical publication times.",
        },
        "effective_issue_interval": {
            "first_issue": starts["forecast_start"].iloc[0].isoformat(),
            "last_issue": starts["forecast_start"].iloc[-1].isoformat(), "n_forecast_starts": len(starts),
        },
        "matched_evaluation": {
            "first_issue": matched["forecast_start"].iloc[0].isoformat(),
            "last_issue": matched["forecast_start"].iloc[-1].isoformat(),
            "n_forecast_starts": len(matched), "n_target_rows": len(matched) * 24,
            "mask": "Complete 2016-hour measured and predicted temperature context plus complete 24-hour future trajectory, shared by perfect-weather and predicted-weather configurations.",
        },
        "rows": {
            "all_output_rows_including_context": len(output), "full_year_candidate_starts": len(candidates),
            "forecast_rows": 7008, "missing_forecast_rows_before_archive": len(output) - 7008,
        },
        "request_records": requests,
    }
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output_path} (sha256={metadata['output_sha256']})")
    print(f"Saved {args.metadata_path}")


if __name__ == "__main__":
    main()
