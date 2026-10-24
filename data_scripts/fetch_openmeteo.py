"""
fetch_openmeteo.py

Downloads hourly cloud cover and PM2.5 (smog) from the Open-Meteo APIs for the
same 5 Pakistani cities as fetch_nasa_power.py, then merges them with the
cleaned NASA POWER data on city + timestamp.

Run from the repo root (after fetch_nasa_power.py and clean_nasa.py):
    python data_scripts/fetch_openmeteo.py

Output:
    data/raw/openmeteo/<city>_<year>.csv           (ignored by git)
    data/processed/merged_weather.csv              (ignored by git, keeps the city column for EDA)
    data/processed/merged_weather_encoded.csv      (ignored by git, city one hot encoded for modelling)
    data_scripts/logs/fetch_openmeteo.log          (commit this as collection evidence)
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

# Settings

# Same cities and coordinates as fetch_nasa_power.py
CITIES = {
    "Lahore": (31.52, 74.36),
    "Karachi": (24.86, 67.00),
    "Islamabad": (33.68, 73.05),
    "Peshawar": (34.02, 71.58),
    "Multan": (30.16, 71.52),
}

YEAR = 2024

# NASA times were converted from UTC to Pakistan time, so the last hours of
# December 2024 fall on 1 Jan 2025. Ask for one extra day so the merge finds them.
START_DATE = f"{YEAR}-01-01"
END_DATE = f"{YEAR + 1}-01-01"

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# cloud_cover: percent of the sky covered by cloud (0 to 100)
# pm2_5: fine particles in the air in ug/m3 (the smog effect)
WEATHER_VARS = "cloud_cover"
AIR_VARS = "pm2_5"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "openmeteo"
PROCESSED_DIR = ROOT / "data" / "processed"
LOG_DIR = ROOT / "data_scripts" / "logs"
NASA_CLEAN = PROCESSED_DIR / "nasa_clean.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Longest gap (in hours) we are willing to fill by interpolation
MAX_GAP_HOURS = 3

# Log to the terminal and to a file at the same time
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fetch_openmeteo.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def fetch_hourly(url, lat, lon, variables, retries=3):
    """Request a full year of hourly data for one location. Returns a DataFrame or None."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": variables,
        "timezone": "Asia/Karachi",  # same clock as the NASA data
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            hourly = response.json()["hourly"]
            df = pd.DataFrame(hourly)
            df["time"] = pd.to_datetime(df["time"])
            return df.rename(columns={"time": "timestamp"})
        except (requests.RequestException, KeyError, ValueError) as error:
            log.warning("Attempt %d failed: %s", attempt, error)
            time.sleep(5 * attempt)  # wait a bit longer after each failure

    return None


def download_city(city, lat, lon):
    """Download cloud cover and PM2.5 for one city and combine them into one table."""
    weather = fetch_hourly(WEATHER_URL, lat, lon, WEATHER_VARS)
    time.sleep(1)  # small pause so we don't overload the API
    air = fetch_hourly(AIR_URL, lat, lon, AIR_VARS)
    time.sleep(1)

    if weather is None or air is None:
        log.error("%s: FAILED (weather ok: %s, air quality ok: %s)",
                  city, weather is not None, air is not None)
        return None

    log.info("%s: %d weather rows, %d air quality rows", city, len(weather), len(air))

    df = weather.merge(air, on="timestamp", how="outer").sort_values("timestamp")
    df.insert(1, "city", city)
    df = df.rename(columns={"pm2_5": "pm25"})
    return df


def main():
    log.info("Starting Open-Meteo download: %d cities, year %d", len(CITIES), YEAR)
    frames = []

    for city, (lat, lon) in CITIES.items():
        df = download_city(city, lat, lon)
        if df is None:
            continue

        out_path = RAW_DIR / f"{city.lower()}_{YEAR}.csv"
        df.to_csv(out_path, index=False)
        log.info("%s: saved %d rows to %s", city, len(df), out_path.relative_to(ROOT))
        frames.append(df)

    if not frames:
        raise RuntimeError("Nothing downloaded from Open-Meteo, check the log.")

    openmeteo = pd.concat(frames, ignore_index=True)

    # Merge with the cleaned NASA data

    if not NASA_CLEAN.exists():
        raise FileNotFoundError(
            f"{NASA_CLEAN} not found. Run fetch_nasa_power.py and clean_nasa.py first, "
            "or download nasa_clean.csv from the link in the README."
        )

    nasa = pd.read_csv(NASA_CLEAN)
    nasa["timestamp"] = pd.to_datetime(nasa["timestamp"])
    log.info("Loaded NASA clean data: %d rows", len(nasa))

    # Left merge: keep every NASA row, add cloud cover and PM2.5 where they match
    merged = nasa.merge(openmeteo, on=["city", "timestamp"], how="left")
    log.info("After merge: %d rows", len(merged))

    # Basic cleaning of the new columns
    merged.loc[~merged["cloud_cover"].between(0, 100), "cloud_cover"] = float("nan")
    merged.loc[merged["pm25"] < 0, "pm25"] = float("nan")

    missing_before = merged[["cloud_cover", "pm25"]].isna().sum().to_dict()
    log.info("Missing values after merge: %s", missing_before)

    # Fill short gaps one city at a time so cities don't mix
    parts = []
    for city, group in merged.groupby("city"):
        group = group.sort_values("timestamp").set_index("timestamp")
        group[["cloud_cover", "pm25"]] = group[["cloud_cover", "pm25"]].interpolate(
            method="time", limit=MAX_GAP_HOURS, limit_area="inside"
        )
        parts.append(group.reset_index())
    merged = pd.concat(parts, ignore_index=True)

    before = len(merged)
    merged = merged.dropna(subset=["cloud_cover", "pm25"])
    log.info("Dropped %d rows still missing cloud cover or PM2.5", before - len(merged))

    merged = merged.sort_values(["city", "timestamp"]).reset_index(drop=True)

    # Save the table with city names first (eda.py needs them)
    merged_path = PROCESSED_DIR / "merged_weather.csv"
    merged.to_csv(merged_path, index=False)
    log.info("Saved %d rows to %s", len(merged), merged_path.relative_to(ROOT))

    # One hot encoding for city, right after the merge and before the model file
    # is saved (requested by Member 2, who one-hot encoded the Kaggle inverter ID)
    merged = pd.get_dummies(merged, columns=["city"], prefix="city", dtype=int)

    encoded_path = PROCESSED_DIR / "merged_weather_encoded.csv"
    merged.to_csv(encoded_path, index=False)
    log.info("Saved %d rows, %d columns to %s",
            len(merged), merged.shape[1], encoded_path.relative_to(ROOT))
    log.info("Finished.")


if __name__ == "__main__":
    main()