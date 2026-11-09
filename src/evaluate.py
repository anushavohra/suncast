import os
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             precision_score, recall_score, f1_score, roc_auc_score)

from model import load_split
from nn_model import SolarNet, SCALE


def get_predictions():
    """Returns (y_train, y_test, preds dict), preds aligned with y_test."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_split()

    gb = joblib.load("models/solar_model.joblib")
    gb_pred = gb.predict(X_test)

    net = SolarNet(X_test.shape[1], dropout=0.0)
    net.load_state_dict(torch.load("models/solar_nn.pt"))
    net.eval()
    Xte = torch.tensor(X_test.values.astype("float32"))
    with torch.no_grad():
        nn_pred = net(Xte).numpy().ravel() * SCALE

    preds = {"Gradient boosting": gb_pred, "Neural network": nn_pred}
    return y_train, y_test, preds


def regression_scores(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    y_train, y_test, preds = get_predictions()

    # high vs low output: threshold = median of TRAIN set, high = strictly above
    threshold = float(np.median(y_train.values))
    y_true_bin = (y_test.values > threshold).astype(int)
    print(f"Threshold (train median DC_POWER): {threshold:.2f} kW")

    rows = []
    for name, p in preds.items():
        mae, rmse, r2 = regression_scores(y_test.values, p)
        y_pred_bin = (p > threshold).astype(int)
        rows.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "Precision": round(precision_score(y_true_bin, y_pred_bin), 4),
            "Recall": round(recall_score(y_true_bin, y_pred_bin), 4),
            "F1": round(f1_score(y_true_bin, y_pred_bin), 4),
            "AUC_ROC": round(roc_auc_score(y_true_bin, p), 4),
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv("results/test_metrics.csv", index=False)
    print("Saved: results/test_metrics.csv")

    # predicted vs actual
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    lim = max(y_test.max(), max(p.max() for p in preds.values()))
    for ax, (name, p) in zip(axes, preds.items()):
        ax.scatter(y_test.values, p, s=3, alpha=0.3)
        ax.plot([0, lim], [0, lim], "r--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Actual DC_POWER (kW)")
        ax.set_ylabel("Predicted DC_POWER (kW)")
    plt.tight_layout()
    plt.savefig("figures/predicted_vs_actual.png", dpi=200, bbox_inches="tight")
    print("Saved: figures/predicted_vs_actual.png")