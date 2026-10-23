# Dataset: Solar Power Generation Data (Kaggle)
# Link: (https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
import pandas as pd

# Load the plant generation data
gen = pd.read_csv("data/raw/Plant_1_Generation_Data.csv")

# Load the weather sensor data
weather = pd.read_csv("data/raw/Plant_1_Weather_Sensor_Data.csv")

print("Generation data shape (rows, columns):", gen.shape)
print(gen.head())

print("Weather data shape (rows, columns):", weather.shape)
print(weather.head())
