"""
split_data.py

Creates the 70/15/15 train/validation/test split used by every SunCast model,
saves which rows belong to each split, and checks that it matches load_split()
in model.py, so all models (and the physics baseline) are tested on exactly
the same rows.

Run from the repo root:
    python src/split_data.py

Output:
    data/processed/split_indices.csv   (row number and split name, ignored by git)
    results/split_summary.csv          (size and target statistics of each split)
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from model import load_split

FEATURES_PATH = "data/processed/generation_features.csv"
SPLIT_PATH = "data/processed/split_indices.csv"
SUMMARY_PATH = "results/split_summary.csv"
TARGET = "DC_POWER"
SEED = 42


def make_split(n_rows):
    """Split row numbers 70/15/15 the same way load_split() does.

    First 30% is held out, then that 30% is cut in half into validation and test.
    The shuffle only depends on the number of rows and the seed, so splitting
    row numbers gives the same rows as splitting the full table.
    """
    rows = pd.RangeIndex(n_rows)
    train_idx, temp_idx = train_test_split(rows, test_size=0.30, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SEED)
    return train_idx, val_idx, test_idx


def check_against_model(train_idx, val_idx, test_idx):
    """Make sure our split is identical to the one model.py trains on."""
    X_train, X_val, X_test, _, _, _ = load_split()
    same = (
        list(X_train.index) == list(train_idx)
        and list(X_val.index) == list(val_idx)
        and list(X_test.index) == list(test_idx)
    )
    if not same:
        raise RuntimeError("Split does not match load_split() in model.py")
    print("Check passed: split is identical to load_split() in model.py")


def main():
    data = pd.read_csv(FEATURES_PATH)
    n = len(data)
    train_idx, val_idx, test_idx = make_split(n)

    # Sanity checks: no row in two splits, and no row left out
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == n

    check_against_model(train_idx, val_idx, test_idx)

    # Save which split each row belongs to
    split = pd.Series("train", index=data.index)
    split[val_idx] = "validation"
    split[test_idx] = "test"
    split_df = pd.DataFrame({"row": data.index, "split": split.values})
    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    split_df.to_csv(SPLIT_PATH, index=False)

    # Summary: each split should look like the full data
    summary = []
    for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        y = data.loc[idx, TARGET]
        summary.append({
            "split": name,
            "rows": len(idx),
            "share_percent": round(100 * len(idx) / n, 1),
            "mean_dc_power": round(y.mean(), 2),
            "std_dc_power": round(y.std(), 2),
            "zero_output_percent": round(100 * (y == 0).mean(), 1),
        })
    summary_df = pd.DataFrame(summary)

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print(f"Total rows: {n}")
    print(summary_df.to_string(index=False))
    print(f"Saved: {SPLIT_PATH}")
    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
