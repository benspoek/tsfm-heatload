# Third-Party Notices

The MIT License in `LICENSE` applies only to original project code,
project-authored documentation, and the synthetic files in `tests/fixtures/`.
It does not relicense datasets or other third-party material. The materials
below remain subject to their respective licenses and attribution
requirements.

## Flensburg District-Heating Data

**Affected paths**

- `flensburg/2020-2024 Stadtwerke Flensburg Heat Network Data Hourly.xlsx`
- `flensburg/demand/heat/heat_dh.csv`
- `flensburg/demand/heat/heat_dh_metadata.json`

**Source and attribution**

Jonas Freißmann, Malte Fritz, Ilja Tuschy, and Stadtwerke Flensburg GmbH,
*Network Data of the District Heating System for the city of Flensburg from
2020-2024*, version 1.0.0.

Source: https://doi.org/10.5281/zenodo.17177421

**License**

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0):
https://creativecommons.org/licenses/by-nc-sa/4.0/

A copy of the license is provided in `LICENSES/CC-BY-NC-SA-4.0.txt`.
Commercial use is not permitted under this license. Shared adaptations must
comply with its attribution and ShareAlike requirements.

**Modifications**

The source workbook is included under its original filename. The derived
`heat_dh.csv` selects the total heat-output series
`STWFL.0FH00W801 / HKW Waermeleistung Gesamt`, removes unrelated source
columns, renames the timestamp and heat fields, converts the timestamps to an
hourly `Europe/Berlin` series, and inserts and interpolates five missing hourly
rows. Processing details are recorded in
`flensburg/demand/heat/heat_dh_metadata.json`.

## Deutscher Wetterdienst Temperature Data

**Affected paths**

- `flensburg/weather/flensburg_weather_temperature.csv`
- `flensburg/weather/flensburg_weather_temperature_metadata.json`
- `flensburg/weather/representative_weeks_2024.csv`

**Source and attribution**

Source: Deutscher Wetterdienst (DWD), Climate Data Center, hourly historical
air-temperature observations:
https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/

The selected station is Glücksburg-Meierwik, station ID `01666`.

**License**

Creative Commons Attribution 4.0 International (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/

A copy of the license is provided in `LICENSES/CC-BY-4.0.txt`.

**Modifications**

The temperature extract selects one station and one temperature variable,
converts timestamps from UTC to `Europe/Berlin`, aligns the observations to
the hourly Flensburg heat-data index for 2020-2024, and writes a reduced CSV.
The representative-week file is derived from its 2024 values by considering
complete Monday-to-Sunday ISO weeks and selecting the hottest week by mean
temperature, the coldest week by mean temperature, and the week with the
largest temperature range. Processing details are recorded in the adjacent
metadata file and `scripts/select_flensburg_representative_weeks.py`.

## Software Dependencies

Python packages listed in `requirements.txt` are dependencies and are not
vendored in this repository. Each installed package remains governed by its
own license. The MIT License for this project does not replace or modify those
licenses.
