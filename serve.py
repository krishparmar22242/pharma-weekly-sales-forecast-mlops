# ======================================================
# 💊 Pharma Weekly Forecast — FastAPI + MLflow (DagsHub Auto Model)
# ======================================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import mlflow.pyfunc
import pandas as pd
import os

# ---- STEP 1: MLflow + DagsHub Integration ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

if MLFLOW_TRACKING_URI:
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
if MLFLOW_TRACKING_USERNAME:
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
if MLFLOW_TRACKING_PASSWORD:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

# ---- STEP 2: Load latest model from DagsHub ----
print("📦 Fetching latest production model from DagsHub...")
try:
    model = mlflow.pyfunc.load_model("models:/PharmaForecast/Production")
    MODEL_SOURCE = "DagsHub Registry"
except Exception as e:
    print("⚠️ No registered model found, falling back to local model...")
    from joblib import load
    model = load("best_model_xgb.pkl")
    MODEL_SOURCE = "Local backup"

print(f"✅ Model loaded successfully from {MODEL_SOURCE}")

# ---- STEP 3: Create FastAPI App ----
app = FastAPI(title="Pharma Weekly Forecast API", version="2.0")

@app.get("/")
def home():
    return {"status": "API running", "model_source": MODEL_SOURCE}

# ---- STEP 4: Prediction Endpoint ----
@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()

        # Expected input features
        expected_features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'rolling_mean_4']
        df = pd.DataFrame([data], columns=expected_features)

        # Predict
        prediction = float(model.predict(df)[0])
        return JSONResponse({"forecast": prediction})
    
    except Exception as e:
        return JSONResponse({"detail": f"Prediction failed: {str(e)}"}, status_code=500)
