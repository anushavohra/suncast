"""Turn a weather forecast into hourly solar power (kW) for a rooftop system.

Input  weather: DataFrame with columns
    time            : hourly timestamps (local time)
    temperature_c   : air temperature in Celsius
    radiation_wm2   : shortwave radiation in W/m2
Output: DataFrame with columns time and kw.
"""

import os
import joblib
import numpy as np
import pandas as pd

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
NOCT_FACTOR = 0.03125  # module temperature = air temperature + 0.03125 * radiation (W/m2)
LOSS_FACTOR = 0.86     # wiring, dust and inverter losses (stated assumption)
_cache = {}

# Also accept the column names used by Open-Meteo
ALIASES = {
    "datetime": "time",
    "temperature_2m": "temperature_c",
    "shortwave_radiation": "radiation_wm2",
    "irradiance_wm2": "radiation_wm2",
}


def _load():
    """Load the model and the scaler once."""
    if not _cache:
        _cache["model"] = joblib.load(os.path.join(MODELS, "solar_model.joblib"))
        _cache["pre"] = joblib.load(os.path.join(MODELS, "scaler.joblib"))
    return _cache


def _predict_plant(hour, month, ambient, module, irradiation):
    """Plant power from the model, averaged over all 22 inverters."""
    items = _load()
    model, pre = items["model"], items["pre"]
    feature_cols = pre["feature_cols"]
    inverter_cols = [c for c in feature_cols if c.startswith("inverter_")]
    num = pre["numeric_cols"]

    raw = pd.DataFrame({
        "AMBIENT_TEMPERATURE": ambient,
        "MODULE_TEMPERATURE": module,
        "IRRADIATION": irradiation,
    })
    X = pd.DataFrame(pre["scaler"].transform(raw[num]), columns=num)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)

    preds = []
    for col in inverter_cols:
        Xi = X.copy()
        for c in inverter_cols:
            Xi[c] = 0.0
        Xi[col] = 1.0
        preds.append(model.predict(Xi[feature_cols]))
    return np.mean(preds, axis=0)


def predict_hourly(weather, system_kw):
    """Hourly kW (numpy array, same order as weather) of a rooftop of size system_kw."""
    w = weather.reset_index(drop=True)
    times = pd.to_datetime(w["time"])
    sun = w["radiation"].astype(float).clip(lower=0).to_numpy()
    air = w["temp_c"].astype(float).to_numpy()
    hour = times.dt.hour.to_numpy()
    month = times.dt.month.to_numpy()

    module = air + NOCT_FACTOR * sun
    actual = _predict_plant(hour, month, air, module, sun / 1000.0)

    full = np.full(len(w), 25.0)
    reference = _predict_plant(hour, month, full, full, np.ones(len(w)))
    fraction = np.where(reference > 1.0, actual / reference, 0.0)
    fraction = np.clip(fraction, 0.0, 1.0)
    fraction[sun <= 0] = 0.0

    return np.clip(fraction * system_kw * LOSS_FACTOR, 0.0, None)


def daily_kwh(hourly):
    """Energy per day in kWh (each row is one hour, so kW x 1 h = kWh)."""
    d = hourly.copy()
    d["date"] = d["time"].dt.date.astype(str)
    return d.groupby("date", as_index=False)["kw"].sum().rename(columns={"kw": "kwh"})


if __name__ == "__main__":
    hours = pd.date_range("2026-11-18 00:00", periods=24, freq="h")
    sun = [max(0.0, 700 * np.sin(np.pi * (h - 6.5) / 11)) for h in range(24)]
    temp = [12 + 10 * max(0.0, np.sin(np.pi * (h - 6) / 12)) for h in range(24)]
    weather = pd.DataFrame({"time": hours, "temp_c": temp, "radiation": sun})
    kw = predict_hourly(weather, system_kw=5)
    print(kw.round(2), kw.sum().round(1))