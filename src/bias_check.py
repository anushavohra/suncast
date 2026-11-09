import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import get_predictions


def build_error_table():
    """One long table: a row per (test row, model) with errors and metadata."""
    y_train, y_test, preds = get_predictions()

    clean = pd.read_csv("data/processed/generation_clean.csv")
    meta = clean.loc[y_test.index, ["DATE_TIME", "SOURCE_KEY_x", "IRRADIATION"]].copy()
    meta["DATE_TIME"] = pd.to_datetime(meta["DATE_TIME"])
    meta["hour"] = meta["DATE_TIME"].dt.hour
    meta["month"] = meta["DATE_TIME"].dt.month

    # inverter short id (Inv01..Inv22), sorted by SOURCE_KEY_x
    keys = sorted(clean["SOURCE_KEY_x"].unique())
    key_to_id = {k: f"Inv{i + 1:02d}" for i, k in enumerate(keys)}
    meta["inverter"] = meta["SOURCE_KEY_x"].map(key_to_id)

    meta["irr_band"] = pd.cut(
        meta["IRRADIATION"],
        bins=[-0.001, 0, 0.2, 0.5, 0.8, 10],
        labels=["0 (night)", "0-0.2", "0.2-0.5", "0.5-0.8", ">0.8"],
    )

    frames = []
    for name, p in preds.items():
        d = meta.copy()
        d["Model"] = name
        d["actual"] = y_test.values
        d["pred"] = np.asarray(p)
        d["err"] = d["pred"] - d["actual"]
        d["abs_err"] = d["err"].abs()
        d["sq_err"] = d["err"] ** 2
        frames.append(d)
    return pd.concat(frames), key_to_id


def summarize(df, by):
    g = df.groupby(["Model", by], observed=True).agg(
        n=("abs_err", "size"),
        MAE=("abs_err", "mean"),
        mean_sq=("sq_err", "mean"),
        mean_actual=("actual", "mean"),
        bias=("err", "mean"),
    ).reset_index()
    g["RMSE"] = np.sqrt(g["mean_sq"])
    g = g.drop(columns="mean_sq")
    return g.round(2)


def plot_mae(table, by, fname, title, rot=0):
    pv = table.pivot(index=by, columns="Model", values="MAE")
    ax = pv.plot(kind="bar", figsize=(9, 4.5))
    ax.set_title(title)
    ax.set_xlabel(by)
    ax.set_ylabel("MAE (kW)")
    plt.xticks(rotation=rot)
    plt.tight_layout()
    plt.savefig(f"figures/{fname}", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: figures/" + fname)
    return pv


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    df, key_to_id = build_error_table()

    # inverter id mapping (for the report)
    pd.DataFrame({"inverter": list(key_to_id.values()),
                  "SOURCE_KEY_x": list(key_to_id.keys())}).to_csv(
        "results/inverter_id_map.csv", index=False)

    specs = [
        ("inverter", "error_by_inverter", "Test MAE by inverter", 90),
        ("hour", "error_by_hour", "Test MAE by hour of day", 0),
        ("month", "error_by_month", "Test MAE by month", 0),
        ("irr_band", "error_by_irradiation", "Test MAE by irradiation band", 0),
    ]
    for by, base, title, rot in specs:
        table = summarize(df, by)
        table.to_csv(f"results/{base}.csv", index=False)
        print(f"\n=== {title} ===")
        pv = plot_mae(table, by, f"{base}.png", title, rot)
        print(pv.round(1).to_string())

    # worst 15 errors per model
    worst = (df.sort_values("abs_err", ascending=False)
               .groupby("Model").head(15)
               [["Model", "DATE_TIME", "inverter", "IRRADIATION",
                 "actual", "pred", "err"]].round(2))
    worst.to_csv("results/worst_errors.csv", index=False)
    print("\n=== Worst errors (top 5 per model) ===")
    print(worst.groupby("Model").head(5).to_string(index=False))
    print("\nSaved: results/worst_errors.csv")