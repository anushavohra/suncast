# SunCast

SunCast is a web app that tells people in Pakistan how much power a rooftop solar system will produce at their home and how much money it will save them. Users enter their city, system size, and electricity tariff, and the app forecasts the next few days of generation from live weather data, then converts it into savings in rupees.

The forecast comes from a machine learning model trained on historical sunlight, weather, and smog data. The app also includes a chatbot that answers common solar questions and suggests solar retailers in the user's city.

## Team

| Member | Role |
|---|---|
| Anoosha Rizwan Vohra (24L-3096) | Repo setup, NASA POWER data collection and cleaning, data split, physics baseline |
| Kainat Afzal (24L-3084) | Solar generation data, feature engineering, models and tuning |
| Manumha Nadeem (24L-3097) | Weather and air quality data, EDA, evaluation, bias check, report |

## Project Structure

```
suncast/
├── data_scripts/         # data collection, cleaning, feature engineering, EDA
│   └── logs/             # logs and cleaning reports (evidence of data collection)
├── data/
│   ├── raw/              # raw downloaded data (not tracked by git)
│   ├── processed/        # cleaned data (not tracked by git)
│   └── retailers.csv     # solar retailers in the 5 cities
├── src/                  # data split, baseline, models, tuning, evaluation, bias check
├── models/               # trained model weights (not tracked by git, see Model Weights)
├── results/              # metrics, tuning tables and bias check results
├── figures/              # charts produced by the scripts
├── report/               # LaTeX source and compiled PDF
│   ├── main.tex
│   ├── sections/
│   └── figures/
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

### Model Weights

Trained models are not stored in this repo. Download them and place them in a `models/` folder in the repo root:

* [Model weights folder](https://drive.google.com/drive/folders/1vSJJs9_Jk7Rc4F996mwIiEv5_G7To9KA?usp=drive_link)
  * `solar_model.joblib`: gradient boosting model (Scikit-Learn)
  * `solar_nn.pt`: neural network (PyTorch)

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

## Running the Models

Run these after the data pipeline, from the repo root:

```bash
python src/split_data.py          # 70/15/15 split, checked against model.py
python src/physics_baseline.py    # fixed formula baseline
python src/model.py               # gradient boosting
python src/tune.py                # hyperparameter search
python src/nn_model.py            # PyTorch neural network
python src/evaluate.py            # test set metrics
python src/bias_check.py          # error by inverter, hour and month
```

To skip training, download the model weights above into `models/` and run `evaluate.py` and `bias_check.py` directly.

## Report

The LaTeX source and compiled PDFs are in the `report/` folder:

* Deliverable 1 (Problem Definition and Data Pipeline): [`report/SunCast_Deliverable_1.pdf`](report/SunCast_Deliverable_1.pdf)
* Deliverable 2 (Model Development and Evaluation): `report/SunCast_Deliverable_2.pdf`, added once the report is final