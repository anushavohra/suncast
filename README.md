# SunCast

SunCast is a web app that tells people in Pakistan how much power a rooftop solar system will produce at their home and how much money it will save them. Users enter their city, system size, and electricity tariff, and the app forecasts the next few days of generation from live weather data, then converts it into savings in rupees.

The forecast comes from a machine learning model trained on historical sunlight, weather, and smog data. The app also includes a chatbot that answers common solar questions and suggests solar retailers in the user's city.

## Team

| Member | Role |
|---|---|
| Anoosha Rizwan Vohra (24L-3096) | Repo setup, NASA POWER data collection and cleaning |
| Kainat Afzal (24L-3084) | Solar generation data, feature engineering |
| Manumha Nadeem (24L-3097) | Weather and air quality data, EDA, report |

## Project Structure

```
suncast/
├── data_scripts/         # data collection, cleaning, feature engineering, EDA
│   └── logs/             # logs and cleaning reports (evidence of data collection)
├── data/
│   ├── raw/              # raw downloaded data (not tracked by git)
│   ├── processed/        # cleaned data (not tracked by git)
│   └── retailers.csv     # solar retailers in the 5 cities
├── report/               # LaTeX source and compiled PDF
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── SunCast_Deliverable_1.pdf
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/anushavohra/suncast.git
cd suncast
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

## Data Sources

* **[NASA POWER API](https://power.larc.nasa.gov/)**: hourly solar irradiance, temperature, humidity, and wind for 5 Pakistani cities (2024)
* **[Open-Meteo API](https://open-meteo.com/)**: hourly cloud cover and air quality (PM2.5)
* **[Kaggle Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)**: real power output and sensor readings from a solar plant
* **Retailer list** (`data/retailers.csv`): 39 solar retailers across the 5 cities, compiled by hand from public Google Maps listings

### Data Downloads

Large data files are not stored in this repo. Cleaned datasets are available here:

* Cleaned NASA POWER data (5 cities, 2024, hourly): [nasa_clean.csv](https://drive.google.com/file/d/1xFconkLj1P9UdFHTOLPeGQy2qwVUD6Ns/view?usp=drive_link)

## Running the Data Pipeline

**Step 1:** Download `Plant_1_Generation_Data.csv` and `Plant_1_Weather_Sensor_Data.csv` from the Kaggle link above and place them in `data/raw/`.

**Step 2:** Run the scripts in this order from the repo root:

```bash
# Weather data
python data_scripts/fetch_nasa_power.py
python data_scripts/clean_nasa.py
python data_scripts/fetch_openmeteo.py

# Solar generation data
python data_scripts/load_kaggle_generation.py
python data_scripts/clean_generation.py
python data_scripts/feature_engineering.py

# Exploratory analysis and charts
python data_scripts/eda.py
```

## Report

The Deliverable 1 report (Problem Definition and Data Pipeline) is in [`report/SunCast_Deliverable_1.pdf`](report/SunCast_Deliverable_1.pdf), with the LaTeX source in the same folder.