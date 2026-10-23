import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the cleaned data
data = pd.read_csv("data/processed/generation_clean.csv")
data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])
print("Rows loaded:", len(data))

# Step 1: Extract hour and month from the timestamp
data["hour"] = data["DATE_TIME"].dt.hour
data["month"] = data["DATE_TIME"].dt.month

# Step 2: Sine/cosine encoding for hour and month
# (hour 23 and hour 0 are close in time, so a cycle is more accurate than a plain number)
data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

# Step 3: One-hot encoding for the inverter ID (a categorical column)
# After the merge, the inverter ID is called SOURCE_KEY_x
data = pd.get_dummies(data, columns=["SOURCE_KEY_x"], prefix="inverter")
print("Columns after one-hot encoding:", data.shape[1])

# Step 4: Scale the numeric input columns (mean = 0, standard deviation = 1)
# DC_POWER and AC_POWER are the targets, so they are NOT scaled
numeric_cols = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
print("Scaled columns:", numeric_cols)

# Remove columns we no longer need
data = data.drop(columns=["DATE_TIME", "PLANT_ID", "SOURCE_KEY_y", "hour", "month",
                          "DAILY_YIELD", "TOTAL_YIELD"])

# Save the final feature table
data.to_csv("data/processed/generation_features.csv", index=False)
print("Final shape (rows, columns):", data.shape)
print("Saved: data/processed/generation_features.csv")
