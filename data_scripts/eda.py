"""
eda.py

Exploratory data analysis for SunCast. Creates the five charts needed for the report:
    1. Sunlight by month per city
    2. Correlation heatmap
    3. Sunlight vs panel output (scatter)
    4. Cloud cover vs sunlight (scatter with trend line)
    5. Missing values chart

Run from the repo root (after fetch_openmeteo.py):
    python data_scripts/eda.py

Needs:
    data/processed/merged_weather.csv       (made by fetch_openmeteo.py)
    data/processed/generation_clean.csv     (made by clean_generation.py, for chart 3)

Output:
    figures/*.png   (commit these, they go into the LaTeX report)
    figures/eda_summary.txt
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MERGED_PATH = PROCESSED_DIR / "merged_weather.csv"
GENERATION_PATH = PROCESSED_DIR / "generation_clean.csv"

YEAR = 2024
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NASA_MISSING_CODE = -999

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="talk")
summary_lines = []


def note(text):
    """Print a finding and keep it for eda_summary.txt (handy for the report)."""
    log.info(text)
    summary_lines.append(text)


def save(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    log.info("Saved %s", path.relative_to(ROOT))


# Chart 1: sunlight by month per city

def chart_sunlight_by_month(df):
    """Average daily sunlight energy (kWh/m2/day) for each month and city."""
    d = df[df["timestamp"].dt.year == YEAR].copy()
    d["date"] = d["timestamp"].dt.date
    d["month"] = d["timestamp"].dt.month

    # Hourly W/m2 added up over a day gives Wh/m2/day, divide by 1000 for kWh
    daily = d.groupby(["city", "month", "date"])["irradiance"].sum().div(1000).reset_index()
    monthly = daily.groupby(["city", "month"])["irradiance"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=monthly, x="month", y="irradiance", hue="city", marker="o",
                 linewidth=2.5, ax=ax)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS)
    ax.set_xlabel("Month (2024)")
    ax.set_ylabel("Average daily sunlight (kWh/m² per day)")
    ax.set_title("Sunlight by month for five Pakistani cities")
    ax.legend(title="City", loc="best", fontsize=12)
    save(fig, "01_sunlight_by_month.png")

    best = monthly.loc[monthly.groupby("city")["irradiance"].idxmax()]
    worst = monthly.loc[monthly.groupby("city")["irradiance"].idxmin()]
    for _, row in best.iterrows():
        low = worst[worst["city"] == row["city"]].iloc[0]
        note(f"{row['city']}: sunniest month {MONTHS[int(row['month']) - 1]} "
             f"({row['irradiance']:.2f} kWh/m2/day), dullest {MONTHS[int(low['month']) - 1]} "
             f"({low['irradiance']:.2f}).")
    yearly = monthly.groupby("city")["irradiance"].mean().sort_values(ascending=False)
    note("Yearly average sunlight (kWh/m2/day): "
         + ", ".join(f"{c} {v:.2f}" for c, v in yearly.items()))


# Chart 2: correlation heatmap

def chart_correlation(df):
    """Correlation between the weather columns, using daytime hours only."""
    cols = ["irradiance", "temp_c", "humidity", "wind_speed", "cloud_cover", "pm25"]
    labels = {
        "irradiance": "Sunlight",
        "temp_c": "Temperature",
        "humidity": "Humidity",
        "wind_speed": "Wind speed",
        "cloud_cover": "Cloud cover",
        "pm25": "PM2.5 (smog)",
    }
    day = df[df["irradiance"] > 0]  # at night sunlight is always 0, which would distort the picture
    corr = day[cols].corr().rename(index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation between weather features (daytime)")
    save(fig, "02_correlation_heatmap.png")

    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    top = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(3)
    for (a, b), value in top.items():
        note(f"Strong link: {a} and {b} (r = {value:.2f}).")
    note(f"Sunlight vs cloud cover: r = {corr.loc['Sunlight', 'Cloud cover']:.2f}; "
         f"sunlight vs PM2.5: r = {corr.loc['Sunlight', 'PM2.5 (smog)']:.2f}.")


# Chart 3: sunlight vs panel output

def chart_sunlight_vs_output():
    """Real solar plant data (Kaggle): sunlight vs DC power."""
    if not GENERATION_PATH.exists():
        log.warning("Skipping chart 3: %s not found. Get generation_clean.csv from Member 2.",
                    GENERATION_PATH.relative_to(ROOT))
        return

    gen = pd.read_csv(GENERATION_PATH, usecols=["IRRADIATION", "DC_POWER"])
    sample = gen.sample(n=min(len(gen), 8000), random_state=42)  # keeps the plot readable
    r = gen["IRRADIATION"].corr(gen["DC_POWER"])

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(sample["IRRADIATION"], sample["DC_POWER"], s=10, alpha=0.35, color="#e08a00")
    coef = np.polyfit(gen["IRRADIATION"], gen["DC_POWER"], 1)
    xs = np.linspace(gen["IRRADIATION"].min(), gen["IRRADIATION"].max(), 100)
    ax.plot(xs, np.polyval(coef, xs), color="#1f3b73", linewidth=2.5, label=f"Trend line (r = {r:.2f})")
    ax.set_xlabel("Sunlight reaching the panels (kW/m²)")
    ax.set_ylabel("Panel output, DC power (kW)")
    ax.set_title("More sunlight, more power (Kaggle solar plant data)")
    ax.legend(loc="upper left")
    save(fig, "03_sunlight_vs_output.png")
    note(f"Sunlight vs panel DC power on the Kaggle plant: r = {r:.3f} ({len(gen)} readings).")


# Chart 4: cloud cover vs sunlight

def chart_cloud_vs_sunlight(df):
    """
    Cloud cover vs sunlight around midday. The Kaggle plant data has no cloud column and
    covers another place and time, so we show the link cloud -> sunlight here, and
    chart 3 shows sunlight -> output.
    """
    hours = df["timestamp"].dt.hour
    noon = df[(hours >= 11) & (hours <= 13)].copy()  # similar sun angle, so clouds are the main difference
    noon["cloud_bin"] = (noon["cloud_cover"] // 10 * 10).clip(upper=90).astype(int)
    order = sorted(noon["cloud_bin"].unique())
    labels = [f"{b}-{b + 10}%" if b < 90 else "90-100%" for b in order]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.boxplot(data=noon, x="cloud_bin", y="irradiance", order=order, color="#f2c14e",
                fliersize=1.5, linewidth=1.2, ax=ax)
    means = noon.groupby("cloud_bin")["irradiance"].mean().reindex(order)
    ax.plot(range(len(order)), means.values, color="#1f3b73", linewidth=3, marker="o",
            label="Average")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel("Cloud cover (grouped in 10% bands, all five cities)")
    ax.set_ylabel("Sunlight at the ground (W/m²), 11:00 to 13:00")
    ax.set_title("More cloud cover, less sunlight")
    ax.legend(loc="upper right")
    save(fig, "04_cloud_cover_vs_sunlight.png")

    r = noon["cloud_cover"].corr(noon["irradiance"])
    clear = noon.loc[noon["cloud_cover"] < 10, "irradiance"].mean()
    cloudy = noon.loc[noon["cloud_cover"] >= 90, "irradiance"].mean()
    note(f"Cloud cover vs midday sunlight: r = {r:.2f}.")
    if pd.notna(clear) and pd.notna(cloudy) and clear > 0:
        note(f"Midday sunlight averages {clear:.0f} W/m2 under clear skies (<10% cloud) and "
             f"{cloudy:.0f} W/m2 under overcast skies (>=90%), a drop of {100 * (1 - cloudy / clear):.0f}%.")


# Chart 5: missing values

def count_missing(path, source, missing_code=None):
    """Percent missing for each column of one raw file (or all files matching a pattern)."""
    files = sorted(path.parent.glob(path.name))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop(columns=[c for c in ["timestamp", "city", "DATE_TIME", "PLANT_ID", "SOURCE_KEY"]
                          if c in df.columns])
    if missing_code is not None:
        df = df.mask(df == missing_code)
    result = df.isna().mean().mul(100).rename("percent").reset_index()
    result.columns = ["column", "percent"]
    result["source"] = source
    return result


def chart_missing_values():
    """Share of missing values per column in each raw dataset, before any cleaning."""
    parts = [
        count_missing(RAW_DIR / "nasa_power" / "*_2024.csv", "NASA POWER", NASA_MISSING_CODE),
        count_missing(RAW_DIR / "openmeteo" / "*_2024.csv", "Open-Meteo"),
        count_missing(RAW_DIR / "Plant_1_Generation_Data.csv", "Kaggle generation"),
        count_missing(RAW_DIR / "Plant_1_Weather_Sensor_Data.csv", "Kaggle weather sensor"),
    ]
    parts = [p for p in parts if p is not None]
    if not parts:
        log.warning("Skipping chart 5: no raw data found in data/raw/")
        return

    miss = pd.concat(parts, ignore_index=True)
    nice = {"ALLSKY_SFC_SW_DWN": "sunlight", "T2M": "temperature", "RH2M": "humidity",
            "WS10M": "wind speed", "cloud_cover": "cloud cover", "pm25": "PM2.5"}
    miss["column"] = miss["column"].replace(nice)
    miss["label"] = miss["source"] + ": " + miss["column"]
    miss = miss.sort_values(["source", "column"]).reset_index(drop=True)

    with plt.rc_context({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12}):
        fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 * len(miss) + 2)))
        palette = dict(zip(miss["source"].unique(),
                           sns.color_palette("Set2", miss["source"].nunique())))
        ax.barh(miss["label"], miss["percent"], color=miss["source"].map(palette))
        top = max(miss["percent"].max(), 1)
        for y, value in enumerate(miss["percent"]):
            ax.text(value + top * 0.01, y, f"{value:.1f}%", va="center", fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, top * 1.15)
        ax.set_xlabel("Missing values (% of rows) in the raw data")
        ax.set_title("Missing values per column before cleaning")
        ax.tick_params(axis="y", labelsize=11)
        if miss["percent"].max() == 0:
            ax.text(0.5, 0.5, "No missing values found", transform=ax.transAxes, ha="center",
                    va="center", fontsize=16, color="#2a7f3f",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#2a7f3f"))
        save(fig, "05_missing_values.png")

    worst = miss.sort_values("percent", ascending=False).iloc[0]
    if worst["percent"] == 0:
        note("Missing values: none found in any raw column.")
    else:
        note(f"Missing values: highest is {worst['label']} at {worst['percent']:.1f}%.")


def main():
    if not MERGED_PATH.exists():
        raise FileNotFoundError(f"{MERGED_PATH} not found. Run fetch_openmeteo.py first.")

    df = pd.read_csv(MERGED_PATH, parse_dates=["timestamp"])
    note(f"Merged weather data: {len(df)} rows, {df['city'].nunique()} cities, "
         f"{df['timestamp'].min():%Y-%m-%d} to {df['timestamp'].max():%Y-%m-%d}.")

    chart_sunlight_by_month(df)
    chart_correlation(df)
    chart_sunlight_vs_output()
    chart_cloud_vs_sunlight(df)
    chart_missing_values()

    (FIG_DIR / "eda_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    log.info("Finished. Charts and eda_summary.txt are in %s", FIG_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()