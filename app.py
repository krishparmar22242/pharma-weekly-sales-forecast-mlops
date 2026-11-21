# ======================================================
# 💊 Pharma Weekly Forecast — Streamlit MLOps Dashboard
# ======================================================

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import mlflow
import os
from mlflow.tracking import MlflowClient

# Streamlit Page Config
st.set_page_config(page_title="Pharma Forecast Dashboard", layout="wide")

# -----------------------
# 🔧 Configuration
# -----------------------
API_URL = "http://127.0.0.1:8000/predict"  # FastAPI endpoint
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

st.title("💊 Pharma Weekly Forecast — MLOps Dashboard")
st.markdown("A unified view of **Training, Prediction & Metrics** powered by **MLflow + DagsHub + FastAPI**")

# -----------------------
# 📈 Section 1: Metrics from MLflow
# -----------------------
st.header("📊 Latest Model Metrics (from MLflow)")

try:
    experiments = client.list_experiments()
    exp_id = experiments[-1].experiment_id  # latest experiment
    runs = client.search_runs(exp_id, order_by=["attributes.start_time DESC"], max_results=1)
    if runs:
        run = runs[0]
        metrics = run.data.metrics
        params = run.data.params
        st.success("✅ Latest experiment fetched from MLflow!")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{metrics.get('MAE', 0):.2f}")
        col2.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
        col3.metric("R²", f"{metrics.get('R2', 0):.3f}")
    else:
        st.warning("No MLflow runs found yet. Train your model first.")
except Exception as e:
    st.error(f"⚠️ Could not fetch metrics from MLflow: {e}")

# -----------------------
# 📉 Section 2: Prediction
# -----------------------
st.header("🧠 Make a Forecast Prediction")

with st.expander("Manual Input for Lag Features"):
    lag_1 = st.number_input("lag_1", value=420.0)
    lag_2 = st.number_input("lag_2", value=415.0)
    lag_3 = st.number_input("lag_3", value=430.0)
    lag_4 = st.number_input("lag_4", value=440.0)
    rolling_mean_4 = st.number_input("rolling_mean_4", value=425.0)

    if st.button("🚀 Predict Weekly Sales"):
        payload = {
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_4": lag_4,
            "rolling_mean_4": rolling_mean_4
        }
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()
            st.success(f"💰 Predicted Weekly Sales: **{result.get('forecast', 'N/A')}**")
        except Exception as e:
            st.error(f"API Error: {e}")

# -----------------------
# 📈 Section 3: Visualization
# -----------------------
st.header("📉 Actual vs Predicted Visualization")

uploaded_file = st.file_uploader("Upload test CSV with actual & predicted columns (optional)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "y" in df.columns and "y_pred" in df.columns:
        st.line_chart(df[["y", "y_pred"]])
        st.write("✅ Visualization complete.")
    else:
        st.warning("CSV must contain columns `y` (actual) and `y_pred` (predicted).")

st.markdown("---")
