"""Matched weather-covariate experiments using fixed, coherent IFS hindcasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import chronos2_full_year_2024 as chronos
import tabpfn_ts_full_year_2024 as tabpfn
from full_year_forecasting_utils import (
    HOURLY_STEP,
    ITEM_ID,
    add_error_columns,
    add_synthetic_model_clock,
    aggregate_to_step,
    forecast_starts_from_rows,
    metric_values,
    parse_timestamp_series,
    read_heat_csv,
    sha256_file,
    to_naive_datetime,
    validate_timeseries_frame,
)
from utils import CHRONOS_PACKAGES, TABPFN_PACKAGES, make_run_id, metadata_envelope, write_metadata


MODES = ("measured_only", "forecast_only", "observed_to_forecast", "dual_history", "dual_forecast_fill")
MODEL_MODES = {"tabpfn": tuple(mode for mode in MODES if mode != "dual_history"), "chronos2": MODES}
SELECTED_MODE = {"tabpfn": "dual_forecast_fill", "chronos2": "dual_history"}
DEFAULT_WEATHER_PATH = Path("flensburg/weather/flensburg_temperature_observed_vs_ecmwf_coherent.csv")
DEFAULT_METADATA_PATH = DEFAULT_WEATHER_PATH.with_name(DEFAULT_WEATHER_PATH.stem + "_metadata.json")
PROVENANCE_COLUMNS = (
    "forecast_issue_time_local", "forecast_run_initialization_utc", "forecast_lead_time_hours"
)


def covariate_mapping(mode: str) -> tuple[dict[str, str], dict[str, str]]:
    observed, predicted = "temperature_observed", "temperature_predicted"
    if mode == "measured_only":
        return {"temperature": observed}, {"temperature": observed}
    if mode == "forecast_only":
        return {"temperature_predicted": predicted}, {"temperature_predicted": predicted}
    if mode == "observed_to_forecast":
        return {"temperature": observed}, {"temperature": predicted}
    if mode == "dual_history":
        return {observed: observed, predicted: predicted}, {predicted: predicted}
    if mode == "dual_forecast_fill":
        return {observed: observed, predicted: predicted}, {observed: predicted, predicted: predicted}
    raise ValueError(f"Unknown covariate mode: {mode}")


def validate_coherent_weather(weather: pd.DataFrame) -> None:
    """Reject mixed runs, inconsistent leads, and unavailable issue-time inputs."""
    validate_timeseries_frame(weather, ["temperature_observed"], "coherent weather", HOURLY_STEP)
    if not weather["timestamp"].is_monotonic_increasing:
        raise ValueError("Coherent weather timestamps must be chronological.")
    if not np.isfinite(weather["temperature_observed"].to_numpy(dtype=float)).all():
        raise ValueError("Observed temperature contains nonfinite values.")
    predicted = weather["temperature_predicted"].notna()
    if not predicted.any():
        raise ValueError("No retrospective predicted temperatures are available.")
    if not np.isfinite(weather.loc[predicted, "temperature_predicted"].to_numpy(dtype=float)).all():
        raise ValueError("Predicted temperature contains nonfinite values.")
    if weather.loc[predicted, list(PROVENANCE_COLUMNS)].isna().any().any():
        raise ValueError("Predicted temperature is missing run/issue/lead provenance.")
    if weather.loc[~predicted, list(PROVENANCE_COLUMNS)].notna().any().any():
        raise ValueError("Run provenance exists without a predicted temperature.")
    for issue, trajectory in weather.loc[predicted].groupby("forecast_issue_time_local", sort=True):
        issue = pd.Timestamp(issue)
        expected_times = pd.date_range(issue, periods=24, freq="1h")
        if len(trajectory) != 24 or not pd.DatetimeIndex(trajectory["timestamp"]).equals(expected_times):
            raise ValueError(f"Issue {issue} must contain exactly its 24 elapsed hourly targets.")
        runs = trajectory["forecast_run_initialization_utc"].unique()
        if len(runs) != 1:
            raise ValueError(f"Issue {issue} mixes ECMWF run initializations.")
        run = pd.Timestamp(runs[0]).tz_convert("UTC")
        expected_run = issue.tz_convert("UTC").normalize() + pd.Timedelta(hours=12)
        if run != expected_run:
            raise ValueError(f"Issue {issue} does not use its designated 12 UTC run.")
        lag = (issue.tz_convert("UTC") - run) / HOURLY_STEP
        if lag != 11:
            raise ValueError(f"Issue {issue} violates the 11-hour initialization-to-issue policy.")
        expected_leads = (pd.DatetimeIndex(trajectory["timestamp"]).tz_convert("UTC") - run) / HOURLY_STEP
        if not np.array_equal(trajectory["forecast_lead_time_hours"].to_numpy(), np.asarray(expected_leads)):
            raise ValueError(f"Issue {issue} contains inconsistent forecast lead times.")
        if not np.array_equal(np.asarray(expected_leads), np.arange(11, 35)):
            raise ValueError(f"Issue {issue} must use initialization leads 11 through 34 hours.")


def read_coherent_weather(path: Path) -> pd.DataFrame:
    weather = pd.read_csv(path)
    required = {"date", "temperature_observed", "temperature_predicted", *PROVENANCE_COLUMNS}
    missing = sorted(required - set(weather.columns))
    if missing:
        raise ValueError(f"Coherent weather file is missing columns: {missing}")
    weather["timestamp"] = parse_timestamp_series(weather["date"])
    weather["forecast_issue_time_local"] = parse_timestamp_series(weather["forecast_issue_time_local"])
    weather["forecast_run_initialization_utc"] = pd.to_datetime(
        weather["forecast_run_initialization_utc"], utc=True, errors="raise"
    )
    for name in ("temperature_observed", "temperature_predicted", "forecast_lead_time_hours"):
        weather[name] = pd.to_numeric(weather[name], errors="raise")
    weather = weather[["timestamp", "temperature_observed", "temperature_predicted", *PROVENANCE_COLUMNS]].copy()
    validate_coherent_weather(weather)
    return weather


def verify_weather_hash(path: Path, metadata: dict[str, object]) -> None:
    expected = metadata.get("output_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Weather metadata must contain the fixed CSV output_sha256.")
    content = path.read_bytes()
    # Windows Git checkouts may change LF to CRLF without changing CSV content.
    actual = hashlib.sha256(content).hexdigest()
    normalized = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
    if expected not in {actual, normalized}:
        raise ValueError("Coherent weather CSV does not match its fixed-input metadata hash.")


def load_data(heat_path: Path, weather_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    verify_weather_hash(weather_path, metadata)
    weather = read_coherent_weather(weather_path)
    heat = aggregate_to_step(read_heat_csv(heat_path), ["heat"], HOURLY_STEP, label="heat")
    data = weather.merge(heat, on="timestamp", how="left", validate="one_to_one")
    validate_timeseries_frame(data, ["heat", "temperature_observed"], "heat/coherent weather", HOURLY_STEP)
    return add_synthetic_model_clock(data, HOURLY_STEP), metadata


def build_window(
    data: pd.DataFrame, issue: pd.Timestamp, context_hours: int, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = data.index[data["timestamp"].eq(issue)].tolist()
    if len(matches) != 1:
        raise ValueError(f"Forecast issue {issue} must occur exactly once.")
    pos = matches[0]
    if context_hours <= 0 or pos < context_hours or pos + 24 > len(data):
        raise ValueError(f"Incomplete context or horizon at {issue}.")
    context = data.iloc[pos - context_hours:pos]
    future = data.iloc[pos:pos + 24]
    # The same mask is mandatory for all modes, including the measured reference.
    required = ["heat", "temperature_observed", "temperature_predicted", *PROVENANCE_COLUMNS]
    if context[required].isna().any().any() or future[required].isna().any().any():
        raise ValueError(f"Incomplete matched measured/predicted window at {issue}.")
    if not future["forecast_issue_time_local"].eq(issue).all():
        raise ValueError(f"Future rows do not belong to issue {issue}.")
    if not context["forecast_issue_time_local"].lt(issue).all():
        raise ValueError(f"Historical predicted temperatures are not from preceding forecast issues at {issue}.")
    if not context["forecast_run_initialization_utc"].lt(context["forecast_issue_time_local"]).all():
        raise ValueError("Historical run initialization must precede its forecast issue.")
    validate_coherent_weather(future)
    past_map, future_map = covariate_mapping(mode)
    clock = "model_timestamp" if "model_timestamp" in data else "timestamp"
    context_df = pd.DataFrame({
        "item_id": ITEM_ID,
        "timestamp": to_naive_datetime(context[clock]),
        "target": context["heat"],
        **{name: context[source] for name, source in past_map.items()},
    })
    future_df = pd.DataFrame({
        "item_id": ITEM_ID,
        "timestamp": to_naive_datetime(future[clock]),
        **{name: future[source] for name, source in future_map.items()},
    })
    actual_columns = ["timestamp"]
    if clock == "model_timestamp":
        actual_columns.append(clock)
    actual_columns.extend(["heat", "temperature_observed", "temperature_predicted", *PROVENANCE_COLUMNS])
    actual = future[actual_columns].rename(columns={"heat": "actual_heat"}).copy()
    if clock == "model_timestamp":
        actual["model_timestamp"] = to_naive_datetime(actual["model_timestamp"])
    return context_df, future_df, actual


def matched_starts(data: pd.DataFrame, year: int, context_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = forecast_starts_from_rows(data, year, 24, 24, value_columns=["heat", "temperature_observed"])
    keep = []
    for issue in candidates["forecast_start"]:
        try:
            build_window(data, issue, context_hours, "measured_only")
        except ValueError:
            keep.append(False)
        else:
            keep.append(True)
    return candidates, candidates.loc[keep].reset_index(drop=True)


def predict_one(pipeline, model: str, data: pd.DataFrame, issue: pd.Timestamp, context_hours: int, mode: str):
    context, future, actual = build_window(data, issue, context_hours, mode)
    started = time.perf_counter()
    if model == "tabpfn":
        if getattr(pipeline, "max_context_length", 0) < len(context):
            raise ValueError("TabPFN-TS would truncate the requested context.")
        prediction = pipeline.predict_df(context_df=context, future_df=future, quantiles=tabpfn.DEFAULT_QUANTILES)
        predicted = tabpfn.flatten_tabpfn_prediction(prediction, tabpfn.DEFAULT_QUANTILES)
        predicted = predicted.rename(columns={"timestamp": "model_timestamp"})
    elif model == "chronos2":
        prediction = pipeline.predict_df(
            context, future_df=future, prediction_length=24, quantile_levels=chronos.DEFAULT_QUANTILES,
            id_column="item_id", timestamp_column="timestamp", target="target",
        )
        predicted = chronos.flatten_chronos_predictions(prediction, chronos.DEFAULT_QUANTILES)
    else:
        raise ValueError(f"Unknown model: {model}")
    elapsed = time.perf_counter() - started
    predicted["model_timestamp"] = pd.to_datetime(predicted["model_timestamp"], errors="raise")
    forecast = actual.merge(predicted, on="model_timestamp", how="left", validate="one_to_one")
    if forecast["predicted_heat"].isna().any():
        raise ValueError(f"Missing predictions for {model}/{mode} at {issue}.")
    forecast = forecast.drop(columns="model_timestamp")
    label = "TabPFN-TS" if model == "tabpfn" else "Chronos-2"
    forecast.insert(0, "model", label)
    forecast.insert(1, "weather_mode", mode)
    forecast.insert(2, "forecast_start", issue.isoformat())
    forecast.insert(3, "horizon_step", np.arange(1, 25))
    forecast.insert(4, "horizon_minutes", np.arange(1, 25) * 60)
    forecast["prediction_seconds"] = elapsed
    forecast["resolution"] = "hourly"
    forecast["context_hours"] = context_hours
    forecast["prediction_hours"] = 24
    forecast["prediction_steps"] = 24
    forecast["context_rows"] = context_hours
    forecast["requested_context_steps"] = context_hours
    forecast["effective_context_steps"] = context_hours
    forecast = add_error_columns(forecast)
    metrics = {
        "model": label, "weather_mode": mode, "forecast_start": issue.isoformat(),
        "resolution": "hourly", "context_hours": context_hours, "prediction_hours": 24, "prediction_steps": 24,
        "n_context_rows": len(context), "n_forecast_rows": len(forecast), "prediction_seconds": elapsed,
        **metric_values(forecast["actual_heat"], forecast["predicted_heat"]),
    }
    return forecast, metrics


def calculate_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, group in raw.groupby("weather_mode", sort=False):
        seconds = group.groupby("forecast_start")["prediction_seconds"].first()
        rows.append({
            "model": group["model"].iloc[0], "weather_mode": mode, "metric_scope": "all",
            "n_forecast_starts": group["forecast_start"].nunique(), "n_rows": len(group),
            "prediction_seconds_total": seconds.sum(), "prediction_seconds_mean_per_forecast_start": seconds.mean(),
            **metric_values(group["actual_heat"], group["predicted_heat"]),
        })
    return pd.DataFrame(rows)


def parse_args(model: str, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flensburg weather sensitivity using coherent retrospective ECMWF predictions.")
    parser.add_argument("--heat-path", type=Path, default=Path("flensburg/demand/heat/heat_dh.csv"))
    parser.add_argument("--weather-comparison-path", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--weather-metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/ecmwf_weather_sensitivity") / model)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=f"{model}_ecmwf_coherent_weather_sensitivity")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--context-hours", type=int, default=2016)
    parser.add_argument("--prediction-hours", type=int, default=24)
    parser.add_argument("--forecast-every-hours", type=int, default=24)
    parser.add_argument("--covariate-modes", default=None, help=f"Comma-separated modes, or all ({', '.join(MODEL_MODES[model])}). Default: measured_only plus the selected mode for this backbone.")
    parser.add_argument("--mode", "--tabpfn-mode", dest="tabpfn_mode", choices=("LOCAL", "CLIENT"), default="LOCAL")
    parser.add_argument("--model-path", default=chronos.DEFAULT_MODEL_PATH)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--max-context-steps", type=int, default=chronos.DEFAULT_MAX_CONTEXT_STEPS)
    parser.add_argument("--max-forecast-starts", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.context_hours <= 0:
        parser.error("--context-hours must be positive.")
    if args.prediction_hours != 24 or args.forecast_every_hours != 24:
        parser.error("The fixed coherent input requires a 24-hour horizon and 24 elapsed-hour cadence.")
    if args.max_forecast_starts is not None and args.max_forecast_starts <= 0:
        parser.error("--max-forecast-starts must be positive.")
    if model == "chronos2" and args.max_context_steps < args.context_hours:
        parser.error("--max-context-steps must accommodate the complete requested context; truncation would change the matched comparison.")
    args.covariate_modes = (
        list(MODEL_MODES[model]) if args.covariate_modes == "all" else
        [part.strip() for part in args.covariate_modes.split(",")] if args.covariate_modes else
        ["measured_only", SELECTED_MODE[model]]
    )
    if model == "tabpfn" and "dual_history" in args.covariate_modes:
        parser.error("TabPFN-TS drops past-only measured temperature in dual_history. Use dual_forecast_fill to retain both historical channels, or forecast_only for predicted temperature alone.")
    if not args.covariate_modes or len(set(args.covariate_modes)) != len(args.covariate_modes) or any(mode not in MODEL_MODES[model] for mode in args.covariate_modes):
        parser.error(f"--covariate-modes must select distinct modes from {', '.join(MODEL_MODES[model])}, or all.")
    return args


def run(model: str, argv: list[str] | None = None) -> None:
    args = parse_args(model, argv)
    data, weather_metadata = load_data(args.heat_path, args.weather_comparison_path, args.weather_metadata_path)
    candidates, starts = matched_starts(data, args.year, args.context_hours)
    complete_matched_count = len(starts)
    if args.max_forecast_starts is not None:
        starts = starts.head(args.max_forecast_starts)
    if starts.empty:
        raise ValueError("No complete matched heat/measured/predicted windows are available.")
    print(f"Model: {'TabPFN-TS' if model == 'tabpfn' else 'Chronos-2'}")
    print(f"Candidate forecast starts: {len(candidates)}")
    print(f"Complete matched forecast starts: {complete_matched_count}")
    print(f"Selected forecast starts per mode: {len(starts)}")
    print(f"First start: {starts['forecast_start'].iloc[0]}")
    print(f"Last start: {starts['forecast_start'].iloc[-1]}")
    print(f"Covariate modes: {', '.join(args.covariate_modes)}")
    if args.dry_run:
        for mode in args.covariate_modes:
            for issue in starts["forecast_start"]:
                build_window(data, issue, args.context_hours, mode)
        print("Dry run complete. No model loading, API requests, predictions, or outputs.")
        return
    started = time.perf_counter()
    run_id = make_run_id(args.run_id, args.run_name)
    run_dir = args.output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    pipeline = (
        tabpfn.initialize_pipeline(args.tabpfn_mode, args.context_hours) if model == "tabpfn" else
        chronos.initialize_pipeline(args)
    )
    run_dir.mkdir(parents=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    parts, metric_rows = [], []
    for mode in args.covariate_modes:
        for number, issue in enumerate(starts["forecast_start"], start=1):
            forecast, metrics = predict_one(pipeline, model, data, issue, args.context_hours, mode)
            parts.append(forecast)
            metric_rows.append(metrics)
            print(f"[{number}/{len(starts)}] {mode}: {issue}", flush=True)
    raw = pd.concat(parts, ignore_index=True)
    raw.to_csv(run_dir / "raw_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics_per_forecast_start.csv", index=False)
    calculate_summary(raw).to_csv(run_dir / "metrics_summary.csv", index=False)
    metadata = metadata_envelope(
        run_id=run_id, run_name=args.run_name, run_dir=run_dir, script_path=sys.argv[0],
        packages=TABPFN_PACKAGES if model == "tabpfn" else CHRONOS_PACKAGES,
    )
    metadata.update({
        "model": {
            "name": "TabPFN-TS" if model == "tabpfn" else "Chronos-2",
            "implementation": "tabpfn_time_series.TabPFNTSPipeline" if model == "tabpfn" else "chronos.Chronos2Pipeline",
            "model_path": args.model_path if model == "chronos2" else None,
            "tabpfn_mode": args.tabpfn_mode if model == "tabpfn" else None,
            "requested_context_steps": args.context_hours,
            "effective_context_steps": args.context_hours,
            "context_capped": False,
        },
        "inputs": {
            "heat_path": str(args.heat_path), "heat_sha256": sha256_file(args.heat_path),
            "weather_comparison_path": str(args.weather_comparison_path), "weather_comparison_sha256": sha256_file(args.weather_comparison_path),
            "weather_metadata_path": str(args.weather_metadata_path), "weather_metadata_sha256": sha256_file(args.weather_metadata_path),
        },
        "experiment": {
            "year": args.year, "resolution": "hourly", "context_hours": args.context_hours,
            "prediction_hours": 24, "forecast_every_hours": 24,
            "n_candidate_forecast_starts": len(candidates), "n_complete_matched_forecast_starts": complete_matched_count,
            "n_selected_forecast_starts_per_mode": len(starts), "covariate_modes": args.covariate_modes,
            "first_issue": starts["forecast_start"].iloc[0].isoformat(), "last_issue": starts["forecast_start"].iloc[-1].isoformat(),
            "covariate_mappings": {mode: {"past": covariate_mapping(mode)[0], "future": covariate_mapping(mode)[1]} for mode in args.covariate_modes},
            "matched_mask": "Every mode requires complete heat, measured temperature, and predicted temperature throughout the same context and horizon.",
            "future_covariates_exclude_heat_target": True,
            "weather_qualification": "Retrospective IFS hindcasts; initialization times are not observed historical publication times. The 11-hour issue lag applies an assumed availability policy.",
            "tabpfn_past_only_support": "In dual_history the standard TabPFN-TS pipeline drops the measured past-only column. Its selected dual_forecast_fill mode repeats the predicted future values in both temperature columns.",
            "measured_only": "Matched perfect-weather reference; future measured temperature is intentionally available only in this reference mode.",
        },
        "weather_builder_metadata": weather_metadata,
        "total_seconds": time.perf_counter() - started,
    })
    write_metadata(run_dir / "run_metadata.json", metadata)
    print(f"Saved run directory: {run_dir}")
