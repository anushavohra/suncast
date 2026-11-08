import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_split():
    """Load the feature table and split it 70/15/15 (train/validation/test)."""
    data = pd.read_csv("data/processed/generation_features.csv")
    # AC_POWER is almost the same as DC_POWER, so it must not be an input
    X = data.drop(columns=["DC_POWER", "AC_POWER"])
    y = data["DC_POWER"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(learning_rate=0.1, max_depth=None, max_iter=200):
    """Gradient boosting regressor with squared error loss."""
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        random_state=42,
    )


def scores(y_true, y_pred):
    """Return MAE, RMSE and R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_split()
    print("Train/Val/Test sizes:", len(X_train), len(X_val), len(X_test))

    model = build_model()
    model.fit(X_train, y_train)

    mae, rmse, r2 = scores(y_val, model.predict(X_val))
    print(f"Validation MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/solar_model.joblib")
    print("Saved: models/solar_model.joblib")