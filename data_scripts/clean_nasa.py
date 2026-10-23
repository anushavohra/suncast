"""
clean_nasa.py

Cleans the raw NASA POWER data downloaded by fetch_nasa_power.py.

Steps:
    1. Load and combine all city CSVs
    2. Replace NASA's missing value code (-999) with NaN
    3. Remove duplicate rows (same city and timestamp)
    4. Handle outliers and physically impossible values
    5. Fill short gaps by interpolation
    6. Save the cleaned data and a before/after report

Run from the repo root:
    python data_scripts/clean_nasa.py

Output:
    data/processed/nasa_clean.csv             (ignored by git)
    data_scripts/logs/clean_nasa_report.txt   (commit this for the report)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "nasa_power"
OUT_DIR = ROOT / "data" / "processed"
LOG_DIR = ROOT / "data_scripts" / "logs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MISSING_CODE = -999

# Friendlier column names for the rest of the project
RENAME = {
    "ALLSKY_SFC_SW_DWN": "irradiance",
    "T2M": "temp_c",
    "RH2M": "humidity",
    "WS10M": "wind_speed",
}
NUMERIC_COLS = list(RENAME.values())

# Hours (Pakistan time) when the sun is always down, in every season
NIGHT_HOURS = [21, 22, 23, 0, 1, 2, 3, 4]

# Longest gap (in hours) we are willing to fill by interpolation
MAX_GAP_HOURS = 3

report_lines = []


def report(text=""):
    """Print a line and also keep it for the report file."""
    print(text)
    report_lines.append(text)


def summarize(df, title):
    report(f"===== {title} =====")
    report(f"Rows: {len(df)}")
    report(f"Duplicate rows (city + timestamp): {df.duplicated(subset=['city', 'timestamp']).sum()}")
    report("Missing values per column:")
    for col in NUMERIC_COLS:
        report(f"    {col}: {df[col].isna().sum()}")
    report()


def count_iqr_outliers(series):
    """Count values outside 1.5 x IQR. Used for reporting only."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()


def main():
    # 1. Load and combine
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {RAW_DIR}. Run fetch_nasa_power.py first.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.rename(columns=RENAME)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    report(f"Loaded {len(files)} files: {[f.name for f in files]}")
    report()

    # 2. Missing value code to NaN (done before the summary so it counts them)
    df[NUMERIC_COLS] = df[NUMERIC_COLS].replace(MISSING_CODE, np.nan)
    df.loc[df["irradiance"] <= MISSING_CODE, "irradiance"] = np.nan

    summarize(df, "BEFORE CLEANING")

    # 3. Duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["city", "timestamp"], keep="first")
    report(f"Removed {before - len(df)} duplicate rows")

    # 4. Outliers and impossible values
    # Negative sunlight is impossible, so set it to 0
    neg_irr = (df["irradiance"] < 0).sum()
    df.loc[df["irradiance"] < 0, "irradiance"] = 0

    # Sunlight above the solar constant (about 1361 W/m2) cannot reach the ground
    high_irr = (df["irradiance"] > 1361).sum()
    df.loc[df["irradiance"] > 1361, "irradiance"] = np.nan

    # Any sunlight during guaranteed night hours is an error, so set it to 0
    night = df["timestamp"].dt.hour.isin(NIGHT_HOURS)
    night_fixed = (night & (df["irradiance"] != 0)).sum()
    df.loc[night, "irradiance"] = 0

    # Temperature outside a realistic range for Pakistan
    bad_temp = (~df["temp_c"].between(-20, 55) & df["temp_c"].notna()).sum()
    df.loc[~df["temp_c"].between(-20, 55), "temp_c"] = np.nan

    # Humidity must be between 0 and 100 percent
    bad_hum = (~df["humidity"].between(0, 100) & df["humidity"].notna()).sum()
    df["humidity"] = df["humidity"].clip(0, 100)

    # Wind speed cannot be negative, and above 40 m/s is unrealistic here
    bad_wind = (~df["wind_speed"].between(0, 40) & df["wind_speed"].notna()).sum()
    df.loc[~df["wind_speed"].between(0, 40), "wind_speed"] = np.nan

    report("Impossible values handled:")
    report(f"    Negative irradiance set to 0: {neg_irr}")
    report(f"    Irradiance above 1361 set to NaN: {high_irr}")
    report(f"    Nighttime irradiance set to 0: {night_fixed}")
    report(f"    Temperature outside -20 to 55 C set to NaN: {bad_temp}")
    report(f"    Humidity clipped to 0 to 100: {bad_hum}")
    report(f"    Wind speed outside 0 to 40 m/s set to NaN: {bad_wind}")
    report()

    # Statistical outliers are only counted, not removed: heatwaves and
    # windy days are real events the model needs to learn from
    report("Statistical outliers (1.5 x IQR), kept on purpose:")
    for col in ["temp_c", "humidity", "wind_speed"]:
        report(f"    {col}: {count_iqr_outliers(df[col].dropna())}")
    report()

    # 5. Fill short gaps, one city at a time so cities don't mix
    parts = []
    for city, group in df.groupby("city"):
        group = group.set_index("timestamp").sort_index()
        group[NUMERIC_COLS] = group[NUMERIC_COLS].interpolate(
            method="time", limit=MAX_GAP_HOURS, limit_area="inside"
        )
        parts.append(group.reset_index())
    df = pd.concat(parts, ignore_index=True)

    # Rows still missing irradiance (long gaps) are dropped
    before = len(df)
    df = df.dropna(subset=NUMERIC_COLS)
    report(f"Dropped {before - len(df)} rows that still had missing values after interpolation")
    report()

    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    summarize(df, "AFTER CLEANING")

    report("Rows per city:")
    for city, count in df["city"].value_counts().sort_index().items():
        report(f"    {city}: {count}")

    # 6. Save
    out_path = OUT_DIR / "nasa_clean.csv"
    df.to_csv(out_path, index=False)
    report()
    report(f"Saved cleaned data to {out_path.relative_to(ROOT)}")

    (LOG_DIR / "clean_nasa_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()