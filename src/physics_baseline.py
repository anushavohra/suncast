"""
physics_baseline.py

A simple fixed formula baseline that the machine learning models are compared
against. It uses the standard way solar output is estimated without ML:

    power = k * irradiation * (1 + GAMMA * (module_temperature - 25))

Output grows in proportion to sunlight and drops about 0.4% for every degree
the panel is hotter than 25 C (GAMMA = -0.004, a typical value for silicon
panels). The constant k (the size of the plant) is the only number fitted,
and it is fitted once on the training set.

We also report a version without the temperature correction (sunlight only)
to show how much the temperature term helps.

The baseline needs the original, unscaled sunlight and temperature values, so
it reads generation_clean.csv and uses the row numbers saved by split_data.py
to test on exactly the same rows as the ML models.

Run from the repo root (after python src/split_data.py):
    python src/physics_baseline.py

Output:
    results/baseline_metrics.csv
    figures/baseline_predicted_vs_actual.png
"""

import os

import matplotlib
matplotlib.use("Agg")  # save plots to file without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import scores

CLEAN_PATH = "data/processed/generation_clean.csv"
FEATURES_PATH = "data/processed/generation_features.csv"
SPLIT_PATH = "data/processed/split_indices.csv"
METRICS_PATH = "results/baseline_metrics.csv"
FIGURE_PATH = "figures/baseline_predicted_vs_actual.png"

IRR = "IRRADIATION"
TEMP = "MODULE_TEMPERATURE"
TARGET = "DC_POWER"

GAMMA = -0.004      # power change per degree C above 25 C
T_REF = 25.0        # standard test temperature for panels (C)


def physics_input(df, use_temperature=True):
    """Sunlight, optionally corrected for panel temperature."""
    x = df[IRR].to_numpy()
    if use_temperature:
        x = x * (1 + GAMMA * (df[TEMP].to_numpy() - T_REF))
    return x


def fit_k(x, y):
    """Least squares value of k for y = k * x (no intercept)."""
    return float(np.sum(x * y) / np.sum(x * x))


def main():
    clean = pd.read_csv(CLEAN_PATH)
    split = pd.read_csv(SPLIT_PATH)

    # The split row numbers refer to the feature table, so both files must
    # have the same rows in the same order
    n_features = len(pd.read_csv(FEATURES_PATH, usecols=[TARGET]))
    if len(clean) != n_features:
        raise RuntimeError(
            f"generation_clean.csv has {len(clean)} rows but the feature table "
            f"has {n_features}. Rerun feature_engineering.py."
        )
    for col in [IRR, TEMP, TARGET]:
        if col not in clean.columns:
            raise KeyError(f"Column {col} not found. Columns are: {list(clean.columns)}")

    rows = {name: split.loc[split["split"] == name, "row"].to_numpy()
            for name in ["train", "validation", "test"]}
    train = clean.iloc[rows["train"]]

    results = []
    test_predictions = {}

    for name, use_temp in [("Physics baseline (sunlight + temperature)", True),
                           ("Physics baseline (sunlight only)", False)]:
        # Fit k on training data only
        k = fit_k(physics_input(train, use_temp), train[TARGET].to_numpy())

        for split_name in ["validation", "test"]:
            part = clean.iloc[rows[split_name]]
            pred = np.clip(k * physics_input(part, use_temp), 0, None)
            mae, rmse, r2 = scores(part[TARGET], pred)
            results.append({
                "model": name,
                "split": split_name,
                "k": round(k, 2),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2": round(r2, 4),
            })
            if split_name == "test":
                test_predictions[name] = (part[TARGET].to_numpy(), pred)

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    results_df.to_csv(METRICS_PATH, index=False)

    print("Physics baseline results:")
    print(results_df.to_string(index=False))
    print(f"Saved: {METRICS_PATH}")

    # Show the ML test results next to ours, if Member 3's file is there
    if os.path.exists("results/test_metrics.csv"):
        print("\nML model test results (from evaluate.py):")
        print(pd.read_csv("results/test_metrics.csv").to_string(index=False))

    # Predicted vs actual plot for the main baseline on the test set
    actual, pred = test_predictions["Physics baseline (sunlight + temperature)"]
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, pred, s=3, alpha=0.3)
    top = max(actual.max(), pred.max())
    plt.plot([0, top], [0, top], color="black", linewidth=1, label="Perfect prediction")
    plt.xlabel("Actual DC power (kW)")
    plt.ylabel("Predicted DC power (kW)")
    plt.title("Physics baseline: predicted vs actual (test set)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=150)
    print(f"Saved: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
