"""Make figures/sample_predictions.png: 3-day forecast for the 5 cities, model vs physics baseline."""

import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
from predictor import predict_hourly, daily_kwh, LOSS_FACTOR, NOCT_FACTOR

# Use the same coordinates as backend/weather.py if they differ
CITIES = {
    "Lahore": (31.5204, 74.3587),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Peshawar": (34.0151, 71.5249),
    "Multan": (30.1978, 71.4711),
}
SYSTEM_KW = 5


def get_forecast(lat, lon, days=3):
    """Hourly temperature and shortwave radiation from Open-Meteo (Asia/Karachi time)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,shortwave_radiation",
        "forecast_days": days,
        "timezone": "Asia/Karachi",
    }
    for attempt in range(3):
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
            r.raise_for_status()
            h = r.json()["hourly"]
            return pd.DataFrame({
                "time": pd.to_datetime(h["time"]),
                "temp_c": h["temperature_2m"],
                "radiation": h["shortwave_radiation"],
            })
        except Exception as exc:
            print("Retry", attempt + 1, "because:", exc)
            time.sleep(2)
    raise RuntimeError("Could not download the forecast. Check the internet and try again.")


def physics_kw(w, system_kw):
    """Simple physics baseline: sunlight x system size x temperature derating x losses.
    Use the same formula as Member 1's baseline in /predict if it is different."""
    sun = w["radiation"].clip(lower=0)
    module = w["temp_c"] + NOCT_FACTOR * sun
    derate = (1 - 0.004 * (module - 25)).clip(lower=0)  # about -0.4 % power per degree above 25 C
    return system_kw * (sun / 1000.0) * derate * LOSS_FACTOR


fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharey=True)
axes = axes.flatten()
rows = []

for ax, (city, (lat, lon)) in zip(axes, CITIES.items()):
    weather = get_forecast(lat, lon)
    model_kw = pd.DataFrame({"time": weather["time"], "kw": predict_hourly(weather, SYSTEM_KW)})
    base_kw = physics_kw(weather, SYSTEM_KW)

    ax.plot(model_kw["time"], model_kw["kw"], label="Model", color="#1f4e9c", linewidth=2)
    ax.plot(weather["time"], base_kw, label="Physics baseline", color="#e08a00",
            linestyle="--", linewidth=1.6)
    ax.set_title(city)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.grid(alpha=0.3)

    base = pd.DataFrame({"time": weather["time"], "kw": base_kw})
    d_model, d_base = daily_kwh(model_kw), daily_kwh(base)
    for (_, a), (_, b) in zip(d_model.iterrows(), d_base.iterrows()):
        rows.append((city, a["date"], round(a["kwh"], 1), round(b["kwh"], 1)))

axes[5].axis("off")  # only 5 cities, so the sixth panel stays empty
axes[0].legend(loc="upper left")
for ax in axes[::2]:
    ax.set_ylabel("Power (kW)")

fig.suptitle("3-day forecast for a {} kW rooftop system: model vs physics baseline".format(SYSTEM_KW))
fig.tight_layout()

os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", "sample_predictions.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("Saved:", out)
print(pd.DataFrame(rows, columns=["city", "date", "model_kwh", "baseline_kwh"]).to_string(index=False))