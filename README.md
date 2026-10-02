# Systematic Evaluation of TabPFN-TS and Chronos-2 for Zero-Shot Heat Load Forecasting in District Heating Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21511737.svg)](https://doi.org/10.5281/zenodo.21511737)

This repository reproduces the paper’s full-year Flensburg forecasts with TabPFN-TS, Chronos-2, and AutoGluon using the included public heat-load and temperature inputs. It also includes representative-week, weather-sensitivity, and Multi-Resolution Residual-Correction Forecaster scripts.

The folder intentionally excludes paper-writing files, plotting scripts, Slurm wrappers, logs, caches, and proprietary Munich input data.

## Contents

- `scripts/`
  - `tabpfn_ts_full_year_2024.py`: TabPFN-TS full-year rolling forecasts.
  - `chronos2_full_year_2024.py`: Chronos-2 full-year rolling forecasts.
  - `autogluon_full_year_2024.py`: AutoGluon full-year benchmark forecasts.
  - `tabpfn_ts_heat_backtest_extreme_weeks_2024.py`: TabPFN-TS forecasts on selected representative weeks.
  - `chronos2_heat_backtest_extreme_weeks_2024.py`: Chronos-2 forecasts on selected representative weeks.
  - `autogluon_extreme_weeks_benchmark.py`: AutoGluon selected-week benchmark forecasts.
  - `tabpfn_ts_heat_backtest_extreme_weeks_overlapping.py`: TabPFN-TS overlapping selected-week forecasts.
  - `chronos2_heat_backtest_extreme_weeks_overlapping.py`: Chronos-2 overlapping selected-week forecasts.
  - `tabpfn_ts_heat_backtest_relevant_context_2024.py`: TabPFN-TS recent-versus-seasonally-relevant context experiment.
  - `tabpfn_weather_forecast_effect_2024.py`: TabPFN-TS weather sensitivity with coherent ECMWF IFS retrospective temperature predictions.
  - `chronos2_weather_forecast_effect_2024.py`: Chronos-2 weather sensitivity with coherent ECMWF IFS retrospective temperature predictions.
  - `build_flensburg_ecmwf_coherent_predictions.py`: downloads and prepares the fixed Flensburg IFS temperature input.
  - `stacked_residual_full_year_2024.py`: TabPFN-TS MRRC full-year forecast.
  - `chronos2_stacked_residual_full_year_2024.py`: Chronos-2 MRRC full-year forecast.
  - `select_flensburg_representative_weeks.py`: reproducibly selects the three Flensburg representative weeks.
  - `full_year_forecasting_utils.py`, `autogluon_forecasting_utils.py`, `tabpfn_ts_heat_forecast.py`, and `utils.py`: shared data, forecasting, and experiment-runtime helpers.
- `flensburg/`
  - Full Flensburg validation data, including heat demand, observed temperatures, coherent IFS temperature predictions, and the selected representative weeks.
- Munich experiment inputs are deliberately excluded.
  - Munich heat-demand and weather-comparison files are not published.
  - The Munich experiment scripts require explicitly supplied, properly licensed input files.

## Setup

The documented Flensburg runs used Python 3.12.3, as recorded in `.python-version`, on a cluster with CUDA 12.6.3. Create an isolated Python 3.12 environment, then install the PyTorch CUDA 12.6 wheel before the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

## Citation

Citation metadata for this repository are provided in
[`CITATION.cff`](CITATION.cff), and the corresponding Zenodo release metadata
are provided in [`.zenodo.json`](.zenodo.json).

## License

Except where otherwise noted, the original source code, project-authored
documentation, and synthetic test fixtures in this repository are licensed
under the [MIT License](LICENSE).

Datasets and third-party materials are not covered by the MIT License. They
remain subject to their respective licenses and attribution requirements, as
documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), the
[LICENSES](LICENSES/) directory, and the accompanying metadata files.

The software is provided without warranty of any kind.

## Reproduce the Flensburg forecasts

Run these three commands from the repository root, using the included data and the default paper configuration:

```bash
python scripts/tabpfn_ts_full_year_2024.py
python scripts/chronos2_full_year_2024.py
python scripts/autogluon_full_year_2024.py
```

All three runners evaluate 2024 at hourly resolution with a 24-hour horizon and ambient temperature as the sole weather covariate. TabPFN-TS and Chronos-2 use a rolling 12-week context (2,016 hourly observations). AutoGluon is fitted once on 2023 target observations; the input history expands with observations available at each forecast issuance during 2024.

Forecasts are issued every 24 **elapsed** hours, starting at 00:00 Europe/Berlin on 1 January 2024. The local issuance time therefore shifts to 01:00 during daylight-saving time and returns to 00:00 afterwards. Each model produces 366 forecast starts with 24 hourly predictions each, covering 8,784 target hours. AutoGluon writes these predictions for each benchmark model. As in the principal paper benchmark, realized future temperature is supplied under the perfect-weather assumption; future heat-load targets are never supplied as covariates.

Validate the fixed inputs and forecast-start construction without loading models, running predictions, or writing outputs:

```bash
python scripts/tabpfn_ts_full_year_2024.py --dry-run
python scripts/chronos2_full_year_2024.py --dry-run
python scripts/autogluon_full_year_2024.py --dry-run
python -m unittest discover -s tests
```

These checks validate data handling and scheduling; they do not verify GPU prediction accuracy or reproduce numerical paper results by themselves.

The included Flensburg heat-demand series is hourly. It supports hourly experiments, but not the 15-minute stacked-residual experiments; those scripts exit immediately because hourly heat data cannot supply quarter-hour residual targets. Munich scripts require separately supplied, properly licensed inputs; no Munich data are included in this release.

## Representative Weeks

The included Flensburg example-week experiments use representative weeks derived from the included Flensburg 2024 hourly temperature data. These example weeks are not the Munich configuration weeks used in the paper. We do not publish the Munich-derived representative-week file because the underlying Munich district-heating data are private.

Only complete local-time ISO weeks from Monday through Sunday are considered. The hottest and coldest weeks are selected by mean temperature, and the week with the highest temperature fluctuation is selected by temperature range (`maximum - minimum`):

- Hottest: `2024-W36`, starting Monday 2024-09-02.
- Coldest: `2024-W03`, starting Monday 2024-01-15.
- Highest temperature fluctuation: `2024-W17`, starting Monday 2024-04-22.

The selections are stored in `flensburg/weather/representative_weeks_2024.csv` and can be regenerated with:

```bash
python scripts/select_flensburg_representative_weeks.py
```

## Weather sensitivity with coherent IFS predictions

The included `flensburg/weather/flensburg_temperature_observed_vs_ecmwf_coherent.csv` contains observed temperatures and ECMWF IFS HRES retrospective temperature predictions obtained from the [Open-Meteo Single Runs API](https://open-meteo.com/en/docs/single-runs-api). Each 24-hour heat forecast uses one complete temperature trajectory from the preceding 12:00 UTC initialization. For the 2024 issuance schedule, initialization precedes issuance by 11 hours and the trajectory's lead times range from 11 to 34 hours. The archive contains IFS Cycle 49R1 hindcasts; their initialization timestamps do not establish when forecasts were actually published in 2024.

Run the sensitivity comparisons using the included fixed inputs, without downloading data during forecasting:

```bash
python scripts/tabpfn_weather_forecast_effect_2024.py
python scripts/chronos2_weather_forecast_effect_2024.py
```

Each runner compares the measured-temperature reference with the corresponding configuration selected in the revised paper. Both use measured and predicted temperatures as separate historical covariates in the predicted-weather configuration. TabPFN-TS fills both future temperature columns with the identical IFS trajectory; Chronos-2 uses only predicted temperature as a future covariate. Future heat targets are excluded. These are Flensburg sensitivity runs using the paper's method; the manuscript's Munich sensitivity results require the private Munich inputs.

The IFS input contains 292 daily trajectories from 15 March to 31 December 2024. A common completeness mask requires twelve weeks of historical predicted temperatures for every configuration, including the measured reference. This leaves 208 matched starts from 7 June to 31 December, with 4,992 hourly predictions per configuration. Historical predicted temperatures retain the trajectories assigned to their historical issue times. These counts apply to weather sensitivity; the principal full-year runs above still use 366 starts.

Add `--dry-run` to either command to validate all input windows without loading models or writing outputs. Alternative covariate configurations are available through `--covariate-modes`; see `--help` for the supported modes. To obtain the IFS input again, run `python scripts/build_flensburg_ecmwf_coherent_predictions.py` as a separate preparation step. The adjacent metadata file records the source requests, coordinates, run assignments, processing, and checksums.

## Data Sources and Licenses

The Flensburg heat-network source workbook is from Freißmann, Fritz, Tuschy, and Stadtwerke Flensburg GmbH, *Network Data of the District Heating System for the city of Flensburg from 2020-2024*, version 1.0.0, [doi:10.5281/zenodo.17177421](https://doi.org/10.5281/zenodo.17177421). It is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The included `flensburg/demand/heat/heat_dh.csv` is an adapted, hourly, timezone-aware extract; its processing details are recorded in the adjacent metadata file.

The Flensburg temperature series is adapted from the [Deutscher Wetterdienst Climate Data Center](https://opendata.dwd.de/climate_environment/CDC/) and is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source: Deutscher Wetterdienst. The selected station and processing coverage are recorded in `flensburg/weather/flensburg_weather_temperature_metadata.json`.

The IFS temperature predictions are supplied by [Open-Meteo](https://open-meteo.com/) using ECMWF IFS HRES and are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The comparison CSV combines these predictions with the attributed DWD observations and records the selected daily trajectories. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the adjacent IFS metadata file for attribution and processing details.

Most scripts write the same output structure to `outputs/`:

- `raw_predictions.csv`
- `metrics_per_forecast_start.csv`
- `metrics_summary.csv`
- `run_metadata.json`
- `command.txt`

The selected-week scripts require and default to the tracked Flensburg representative-week file; they fail clearly if it is missing or malformed.
