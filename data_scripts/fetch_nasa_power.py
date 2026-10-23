"""
fetch_nasa_power.py

Downloads hourly solar and weather data from the NASA POWER API
for 5 Pakistani cities and saves one raw CSV per city.

Run from the repo root:
    python data_scripts/fetch_nasa_power.py

Output:
    data/raw/nasa_power/<city>_<year>.csv   (ignored by git)
    data_scripts/logs/fetch_nasa_power.log  (commit this as collection evidence)
"""

import calendar
import logging
import time
from pathlib import Path

import pandas as pd
import requests

# Settings

CITIES = {
    "Lahore": (31.52, 74.36),
    "Karachi": (24.86, 67.00),
    "Islamabad": (33.68, 73.05),
    "Peshawar": (34.02, 71.58),
    "Multan": (30.16, 71.52),
}

# ALLSKY_SFC_SW_DWN: solar irradiance reaching the ground (accounts for clouds)
# T2M: air temperature at 2 m (C)
# RH2M: relative humidity at 2 m (%)
# WS10M: wind speed at 10 m (m/s)
PARAMETERS = ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS10M"]

YEAR = 2024

API_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "nasa_power"
LOG_DIR = ROOT / "data_scripts" / "logs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log to the terminal and to a file at the same time
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fetch_nasa_power.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def fetch_month(lat, lon, year, month, retries=3):
    """Request one month of hourly data for one location. Returns the JSON or None."""
    last_day = calendar.monthrange(year, month)[1]
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "RE",  # renewable energy community
        "latitude": lat,
        "longitude": lon,
        "start": f"{year}{month:02d}01",
        "end": f"{year}{month:02d}{last_day}",
        "format": "JSON",
        "time-standard": "utc",
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(API_URL, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            log.warning("Attempt %d failed: %s", attempt, error)
            time.sleep(5 * attempt)  # wait a bit longer after each failure

    return None


def to_dataframe(payload, city):
    """Turn the API response into a table with one row per hour."""
    values = payload["properties"]["parameter"]

    # values looks like {"T2M": {"2024010100": 12.3, ...}, "RH2M": {...}, ...}
    df = pd.DataFrame(values)

    # Index keys are "YYYYMMDDHH" in UTC; convert to Pakistan time
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H", utc=True)
    df.index = df.index.tz_convert("Asia/Karachi").tz_localize(None)
    df.index.name = "timestamp"

    df = df.reset_index()
    df.insert(1, "city", city)
    return df


def main():
    log.info("Starting NASA POWER download: %d cities, year %d", len(CITIES), YEAR)
    summary = {}

    for city, (lat, lon) in CITIES.items():
        frames = []

        for month in range(1, 13):
            payload = fetch_month(lat, lon, YEAR, month)

            if payload is None:
                log.error("%s %d/%02d: FAILED after all retries", city, YEAR, month)
                continue

            df = to_dataframe(payload, city)
            frames.append(df)

            fill_value = payload.get("header", {}).get("fill_value")
            log.info("%s %d/%02d: %d rows (missing value code: %s)",
                     city, YEAR, month, len(df), fill_value)

            time.sleep(1)  # small pause so we don't overload the API

        if not frames:
            log.error("%s: no data downloaded", city)
            continue

        city_df = pd.concat(frames, ignore_index=True)
        out_path = RAW_DIR / f"{city.lower()}_{YEAR}.csv"
        city_df.to_csv(out_path, index=False)

        summary[city] = len(city_df)
        log.info("%s: saved %d rows to %s", city, len(city_df), out_path.relative_to(ROOT))

    log.info("Finished. Rows per city: %s", summary)


if __name__ == "__main__":
    main()