# Heat Load Forecasting Experiment Code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21511738.svg)](https://doi.org/10.5281/zenodo.21511738)

This folder contains the shareable experiment code used for the TabPFN-TS, Chronos-2, AutoGluon, and Multi-Resolution Residual-Correction Forecaster evaluations.

The folder intentionally excludes paper-writing files, plotting scripts, Slurm wrappers, logs, caches, data-scraper/preparation scripts, and unpublished Munich input data.

<!-- TODO(publication): Add scripts/prepare_flensburg_heat.py and
scripts/build_flensburg_weather_temperature.py. Then revise the statement above
and the Contents and usage documentation to describe the reproducible Flensburg
data-preparation workflow instead of saying that all preparation scripts are excluded. -->

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
  - `tabpfn_weather_forecast_effect_2024.py`: TabPFN-TS full-year weather-forecast sensitivity experiment.
  - `chronos2_weather_forecast_effect_2024.py`: Chronos-2 full-year weather-forecast sensitivity experiment.
  - `stacked_residual_full_year_2024.py`: TabPFN-TS MRRC full-year forecast.
  - `chronos2_stacked_residual_full_year_2024.py`: Chronos-2 MRRC full-year forecast.
  - `select_flensburg_representative_weeks.py`: reproducibly selects the three Flensburg representative weeks.
  - `full_year_forecasting_utils.py`, `autogluon_forecasting_utils.py`, `tabpfn_ts_heat_forecast.py`, and `utils.py`: shared data, forecasting, and experiment-runtime helpers.
- `flensburg/`
  - Full Flensburg validation data, including heat demand, weather data, and the selected representative weeks.
- Munich experiment inputs are deliberately excluded.
  - Munich heat-demand and weather-comparison files are not published.
  - The Munich experiment scripts require explicitly supplied, properly licensed input files.

## Setup

The cluster environment uses Python 3.12.3, as recorded in `.python-version`, and CUDA 12.6.3. Install the matching PyTorch CUDA 12.6 wheel before the remaining dependencies:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

The experiment versions are pinned to `tabpfn-time-series==1.1.0`, `tabpfn==8.0.3`, `chronos-forecasting==2.2.2`, and `autogluon.timeseries==1.5.0`. TimesFM is not an environment dependency. Chronos-2 is run only through the dedicated Chronos scripts, while the AutoGluon benchmarks use AutoGluon's tabular, statistical, and optional neural models without custom Chronos-2 or TimesFM wrappers.

## Citation

The archived release is available from Zenodo:
[doi:10.5281/zenodo.21511738](https://doi.org/10.5281/zenodo.21511738).

## License

Except where otherwise noted, the original source code, project-authored
documentation, and synthetic test fixtures in this repository are licensed
under the [MIT License](LICENSE).

Datasets and third-party materials are not covered by the MIT License. They
remain subject to their respective licenses and attribution requirements, as
documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), the
[LICENSES](LICENSES/) directory, and the accompanying metadata files.

The software is provided without warranty of any kind.

## Usage Notes

Run scripts from the root of this folder so the relative data paths resolve correctly:

```bash
python scripts/tabpfn_ts_full_year_2024.py
python scripts/chronos2_full_year_2024.py
python scripts/autogluon_full_year_2024.py
```

For Munich experiments, provide properly licensed local heat-demand and weather files explicitly:

```bash
python scripts/tabpfn_ts_full_year_2024.py \
  --heat-path path/to/private/munich/heat_dh.csv \
  --weather-path path/to/private/munich/munich_weather_with_solar_precipitation.csv
```

For Flensburg validation runs, use the included data:

```bash
python scripts/tabpfn_ts_full_year_2024.py \
  --dataset-name flensburg \
  --heat-path flensburg/demand/heat/heat_dh.csv \
  --weather-path flensburg/weather/flensburg_weather_temperature.csv \
  --weather-columns temperature
```

The included Flensburg heat-demand series is hourly. It supports hourly experiments, but not the 15-minute stacked-residual experiments; those scripts exit immediately because hourly heat data cannot supply quarter-hour residual targets.

## Representative Weeks

The published selected-week experiments use representative weeks derived from the included Flensburg 2024 hourly temperature data. We intentionally do not publish or reuse the Munich-derived representative-week file because the underlying Munich district-heating data are private.

Only complete local-time ISO weeks from Monday through Sunday are considered. The hottest and coldest weeks are selected by mean temperature, and the week with the highest temperature fluctuation is selected by temperature range (`maximum - minimum`):

- Hottest: `2024-W36`, starting Monday 2024-09-02.
- Coldest: `2024-W03`, starting Monday 2024-01-15.
- Highest temperature fluctuation: `2024-W17`, starting Monday 2024-04-22.

The selections are stored in `flensburg/weather/representative_weeks_2024.csv` and can be regenerated with:

```bash
python scripts/select_flensburg_representative_weeks.py
```

## Data Sources and Licenses

The Flensburg heat-network source workbook is from Freißmann, Fritz, Tuschy, and Stadtwerke Flensburg GmbH, *Network Data of the District Heating System for the city of Flensburg from 2020-2024*, version 1.0.0, [doi:10.5281/zenodo.17177421](https://doi.org/10.5281/zenodo.17177421). It is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The included `flensburg/demand/heat/heat_dh.csv` is an adapted, hourly, timezone-aware extract; its processing details are recorded in the adjacent metadata file.

The Flensburg temperature series is adapted from the [Deutscher Wetterdienst Climate Data Center](https://opendata.dwd.de/climate_environment/CDC/) and is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source: Deutscher Wetterdienst. The selected station and processing coverage are recorded in `flensburg/weather/flensburg_weather_temperature_metadata.json`.

Most scripts write the same output structure to `outputs/`:

- `raw_predictions.csv`
- `metrics_per_forecast_start.csv`
- `metrics_summary.csv`
- `run_metadata.json`
- `command.txt`

The selected-week scripts require and default to the tracked Flensburg representative-week file; they fail clearly if it is missing or malformed.
