import os
import pandas as pd

# Load raw data
gen = pd.read_csv("data/raw/Plant_1_Generation_Data.csv")
weather = pd.read_csv("data/raw/Plant_1_Weather_Sensor_Data.csv")
print("Rows at the start:", len(gen))

# Convert both date columns to the same datetime format
gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], format="%d-%m-%Y %H:%M")
weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"])

# Step 1: Check and remove missing values
print("Missing values per column:")
print(gen.isna().sum())
gen = gen.dropna()

# Step 2: Remove duplicate rows (same timestamp and same inverter)
gen = gen.drop_duplicates(subset=["DATE_TIME", "SOURCE_KEY"])
print("Rows after removing duplicates:", len(gen))

# Step 3: Remove negative power values (impossible readings)
gen = gen[(gen["DC_POWER"] >= 0) & (gen["AC_POWER"] >= 0)]

# Merge generation data with weather data on timestamp
data = gen.merge(weather, on=["DATE_TIME", "PLANT_ID"], how="inner")
print("Rows after merging with weather data:", len(data))

# Step 4: Remove inverter fault outliers
# (there is sunlight, but the inverter output is zero)
fault = (data["IRRADIATION"] > 0.2) & (data["DC_POWER"] == 0)
print("Inverter fault rows found:", fault.sum())
data = data[~fault]

# Step 5: Remove invalid night readings
# (sunlight should be zero at night)
hours = data["DATE_TIME"].dt.hour
night = ((hours < 5) | (hours > 19)) & (data["IRRADIATION"] > 0)
print("Invalid night rows found:", night.sum())
data = data[~night]

print("Rows at the end:", len(data))

# Save the cleaned data
os.makedirs("data/processed", exist_ok=True)
data.to_csv("data/processed/generation_clean.csv", index=False)
print("Saved: data/processed/generation_clean.csv")
