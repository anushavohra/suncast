# SunCast

SunCast is a web app that tells people in Pakistan how much power a rooftop solar system will produce at their home and how much money it will save them. Users enter their city, system size, and electricity tariff, and the app forecasts the next few days of generation from live weather data, then converts it into savings in rupees.

The forecast comes from a machine learning model (gradient boosting) trained on real solar plant data and fed with live weather forecasts. The app also includes a chatbot that answers common solar questions and suggests solar retailers in the user's city.

## Live App

* **Web app:** https://suncast-solar.streamlit.app
* **Backend API:** https://suncast-psi.vercel.app (interactive docs at [/docs](https://suncast-psi.vercel.app/docs), status at [/health](https://suncast-psi.vercel.app/health))

## Team

| Member | Role |
|---|---|
| Anoosha Rizwan Vohra (24L-3096) | Repo setup, NASA POWER data collection and cleaning, data split, physics baseline, FastAPI backend and Vercel deployment, live frontend deployment |
| Kainat Afzal (24L-3084) | Solar generation data, feature engineering, models and tuning, inference pipeline (predictor), Gemini chatbot |
| Manumha Nadeem (24L-3097) | Weather and air quality data, EDA, evaluation, bias check, Streamlit frontend, final report |

## How It Works

1. The user picks a city, system size (kW), tariff (Rs per unit) and number of days (1 to 7) in the Streamlit frontend.
2. The frontend sends the request to the FastAPI backend on Vercel.
3. The backend gets the hourly weather forecast for that city from Open-Meteo, estimates the panel temperature, and runs the trained gradient boosting model.
4. The model output is scaled to the user's system size (with a 0.86 loss factor) and turned into kWh per day and savings in rupees. A physics formula is returned alongside it as a comparison line.
5. The chatbot answers questions using the Gemini API, a team written FAQ and the city's retailers. If the API is unavailable, it falls back to the FAQ.

| Endpoint | What it does |
|---|---|
| `GET /health` | Status, which model is in use, list of cities |
| `POST /predict` | Forecast: hourly kW, daily kWh, savings in Rs, physics baseline |
| `GET /retailers?city=Lahore` | Solar shops in a city |
| `POST /chat` | Chatbot reply to a solar question |

## Project Structure

```
suncast/
├── backend/              # FastAPI backend deployed on Vercel
│   ├── app.py            # endpoints and input checks
│   ├── weather.py        # Open-Meteo forecast for the 5 cities
│   ├── physics.py        # physics baseline and fallback formula
│   ├── predictor.py      # forecast weather -> hourly kW with the trained model
│   ├── chatbot.py        # Gemini chatbot with FAQ fallback
│   ├── solar_faq.md      # knowledge used by the chatbot
│   ├── models/           # trained model and scaler used by the app
│   ├── data/             # retailers.csv for the backend
│   └── requirements.txt
├── frontend/             # Streamlit frontend deployed on Streamlit Community Cloud
│   ├── app.py
│   └── requirements.txt
├── data_scripts/         # data collection, cleaning, feature engineering, EDA
│   └── logs/             # logs and cleaning reports (evidence of data collection)
├── data/
│   ├── raw/              # raw downloaded data (not tracked by git)
│   ├── processed/        # cleaned data (not tracked by git)
│   └── retailers.csv     # solar retailers in the 5 cities
├── src/                  # data split, baseline, models, tuning, evaluation, bias check
├── results/              # metrics, tuning tables and bias check results
├── figures/              # charts produced by the scripts
├── report/               # LaTeX source and compiled PDFs
├── .env.example          # settings the app needs (no real keys)
├── requirements.txt      # libraries for the data pipeline and model training
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
* **[Open-Meteo API](https://open-meteo.com/)**: historical cloud cover and air quality (PM2.5), and the live weather forecast used by the app
* **[Kaggle Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)**: real power output and sensor readings from a solar plant
* **Retailer list** (`data/retailers.csv`): 39 solar retailers across the 5 cities, compiled by hand from public Google Maps listings

### Data Downloads

Large data files are not stored in this repo. Cleaned datasets are available here:

* Cleaned NASA POWER data (5 cities, 2024, hourly): [nasa_clean.csv](https://drive.google.com/file/d/1xFconkLj1P9UdFHTOLPeGQy2qwVUD6Ns/view?usp=drive_link)

### Model Weights

The model and scaler used by the live app are in `backend/models/` (small files, under 1 MB). The trained models from Deliverable 2 can also be downloaded here and placed in a `models/` folder in the repo root:

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
python src/split_data.py                 # 70/15/15 split, checked against model.py
python src/physics_baseline.py           # fixed formula baseline
python src/model.py                      # gradient boosting
python src/tune.py                       # hyperparameter search
python src/nn_model.py                   # PyTorch neural network
python src/evaluate.py                   # test set metrics
python src/bias_check.py                 # error by inverter, hour and month
python src/make_sample_predictions.py    # 3 day forecast figure for the 5 cities
```

To skip training, download the model weights above into `models/` and run `evaluate.py` and `bias_check.py` directly.

## Running the App Locally

**Backend** (needs Python 3.10 or newer for the pinned scikit-learn version):

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs to try the endpoints. For the chatbot, set the environment variables listed in `.env.example` first. Without a Gemini key, the chatbot answers from the FAQ.

**Frontend** (in a second terminal, from the repo root):

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

The frontend reads the backend address from Streamlit secrets (`BACKEND_URL`). Without it, the app runs in mock mode with sample data.

## Configuration

| Setting | Where it is set | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Vercel environment variables | API key for the chatbot (never committed) |
| `GEMINI_MODEL` | Vercel environment variables | Gemini model name, currently `gemini-3.5-flash-lite` |
| `BACKEND_URL` | Streamlit secrets | Address of the backend, `https://suncast-psi.vercel.app` |

Both the backend (Vercel) and the frontend (Streamlit Community Cloud) redeploy automatically on every push to `main`.

## Report

The LaTeX source and compiled PDFs are in the `report/` folder:

* Deliverable 1 (Problem Definition and Data Pipeline): [`report/SunCast_Deliverable_1.pdf`](report/SunCast_Deliverable_1.pdf)
* Deliverable 2 (Model Development and Evaluation): [`report/SunCast_Deliverable_2.pdf`](report/SunCast_Deliverable_2.pdf)
* Final report (Deliverable 3): [`report/SunCast_Deliverable_3.pdf`](report/SunCast_Deliverable_3.pdf)