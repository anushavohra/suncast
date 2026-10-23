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
├── data_scripts/     # data collection, cleaning, feature engineering, EDA
├── data/
│   ├── raw/          # raw downloaded data (not tracked by git)
│   └── processed/    # cleaned data (not tracked by git)
├── report/           # LaTeX report
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

* **NASA POWER API**: hourly solar irradiance, temperature, humidity, and wind for Pakistani cities
* **Open-Meteo API**: historical cloud cover and air quality (PM2.5)
* **Kaggle Solar Power Generation Data**: real power output from solar plants
* **Retailer list**: solar retailers by city, compiled by the team

Large data files are not stored in this repo. Download links will be added here.

## Running the Data Pipeline

```bash
python data_scripts/fetch_nasa_power.py
python data_scripts/clean_nasa.py
```

More scripts will be added as the project progresses.