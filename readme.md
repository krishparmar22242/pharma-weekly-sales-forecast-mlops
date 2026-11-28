# 💊 Pharma Weekly Forecast — End-to-End MLOps Pipeline

An end-to-end MLOps pipeline that forecasts weekly pharmaceutical sales using time-series and gradient-boosted models, with full experiment tracking, automated CI/CD, a deployed prediction API, and a monitoring dashboard.

## Overview

This project predicts weekly pharma sales (aggregated across 8 product categories) using historical sales data. It compares four forecasting approaches — Holt-Winters, ARIMA, Prophet, and a lag-feature-enhanced XGBoost model — and automatically tracks, deploys, and monitors the best-performing model.

**Best model:** XGBoost (Lag-Enhanced) — **R² = 0.907**, MAE = 22.48, RMSE = 37.46, outperforming all statistical baselines by a wide margin.

## Architecture

```
Data (salesweekly.csv)
   → Feature Engineering (lag + rolling-mean features)
   → Model Training (4 candidate models)
   → Experiment Tracking (MLflow + DagsHub)
   → Best Model Selection (highest R²)
   → Model Registry (DagsHub)
   → REST API (FastAPI)
   → Dashboard (Streamlit)
   → CI/CD (GitHub Actions: train → deploy → monitor)
```

## Features

- **Multi-model benchmarking** — Holt-Winters, ARIMA(1,1,1), Prophet (tuned), and XGBoost with lag/rolling-mean features, evaluated on a held-out 15% test split.
- **Experiment tracking** — every run's parameters, metrics (MAE/RMSE/R²), and forecast plots logged to MLflow via DagsHub.
- **Automated model export** — script to pull the highest-R² run from the MLflow registry and export it for deployment.
- **Prediction API** — FastAPI service that loads the production model from the DagsHub registry (with local fallback) and serves real-time forecasts.
- **Interactive dashboard** — Streamlit app showing live MLflow metrics, manual prediction input, and actual-vs-predicted visualization.
- **CI/CD pipeline** — GitHub Actions workflow that trains the model, packages it as an artifact, simulates deployment, and runs automated health/drift monitoring on every push to `main`.

## Tech Stack

| Layer | Tools |
|---|---|
| Modeling | XGBoost, Prophet, statsmodels (ARIMA, Holt-Winters) |
| Experiment Tracking | MLflow, DagsHub |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |
| Core | Python, pandas, NumPy, scikit-learn |

## Project Structure

```
.
├── train_script.py          # Data prep, feature engineering, model training, MLflow logging
├── export_model.py          # Pulls best run from MLflow registry, saves as .pkl
├── api.py                   # FastAPI prediction service
├── dashboard.py             # Streamlit MLOps dashboard
├── test_mlops_api.py        # API health + prediction smoke tests
├── requirements.txt
└── .github/workflows/
    └── mlops-pipeline.yml   # CI/CD: build-train-track → deploy → monitor
```

## Setup

1. **Clone the repo and install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables** (required — do not hardcode credentials)
   ```bash
   export MLFLOW_TRACKING_URI="https://dagshub.com/<your-username>/MLOPS.mlflow"
   export MLFLOW_TRACKING_USERNAME="<your-username>"
   export MLFLOW_TRACKING_PASSWORD="<your-dagshub-token>"
   ```

3. **Train the model**
   ```bash
   python train_script.py
   ```

4. **Export the best model from MLflow**
   ```bash
   python export_model.py
   ```

5. **Run the API**
   ```bash
   uvicorn api:app --reload
   ```

6. **Run the dashboard**
   ```bash
   streamlit run dashboard.py
   ```

## API Reference

**`GET /`** — health check, returns API status and active model source.

**`POST /predict`** — returns a weekly sales forecast.

Request body:
```json
{
  "lag_1": 450.5,
  "lag_2": 430.2,
  "lag_3": 400.0,
  "lag_4": 380.5,
  "rolling_mean_4": 415.0
}
```

Response:
```json
{ "forecast": 438.92 }
```

## CI/CD Pipeline

On every push to `main`, GitHub Actions runs a three-stage pipeline:

1. **Build, Train & Track** — installs dependencies, runs `train_script.py`, logs the run to MLflow, uploads the trained model as a build artifact.
2. **Deploy** — downloads the trained model artifact and simulates a FastAPI deployment with a live health check.
3. **Monitor** — checks API health and logs simulated accuracy, latency, and drift metrics to `monitoring.log`.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **XGBoost (Lag-Enhanced)** | **22.48** | **37.46** | **0.907** |
| Prophet (Tuned) | 88.19 | 115.82 | 0.042 |
| ARIMA(1,1,1) | 97.65 | 123.57 | -0.090 |
| Holt-Winters | 107.19 | 131.62 | -0.236 |

The lag-feature-enhanced XGBoost model substantially outperformed all classical time-series baselines, indicating that recent sales history and short-term rolling trends are stronger predictors of weekly pharma sales than seasonal decomposition alone on this dataset.

## Notes

- Credentials (MLflow/DagsHub tokens) must be supplied via environment variables — never commit them to the repository.
- The CI/CD deployment and monitoring stages run against a local, ephemeral FastAPI instance within the GitHub Actions runner for demonstration purposes; they simulate production deployment rather than deploying to a live external host.

## Author

Krish Parmar
