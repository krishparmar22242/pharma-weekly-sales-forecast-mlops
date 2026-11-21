# ======================================================
# 💊 Pharma Weekly Forecast — MLflow + DagsHub Integration (Final)
# ======================================================

import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib
from mlflow.tracking import MlflowClient

# ---- STEP 1: Configure DagsHub MLflow via Environment Variables ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Pharma Weekly Forecast (Final)")

print(f"🔐 Authenticated as {MLFLOW_TRACKING_USERNAME}")
print(f"✅ Connected to MLflow as {MLFLOW_TRACKING_USERNAME}")

# ---- STEP 2: Load & Prepare Data ----
print("🚀 Starting Pharma Weekly Forecast Training...")

df = pd.read_csv("salesweekly.csv")
df.rename(columns={"datum": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

cols_to_sum = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
df["Total_Sales"] = df[cols_to_sum].sum(axis=1)
df = df.sort_values("Date").dropna(subset=["Total_Sales"])
df = df[df["Total_Sales"] > 0].reset_index(drop=True)
df.rename(columns={"Date": "ds", "Total_Sales": "y"}, inplace=True)

print(f"✅ Data ready | {len(df)} weekly records")

def create_lag_features(data, lags=[1, 2, 3, 4]):
    df_lag = data.copy()
    for lag in lags:
        df_lag[f"lag_{lag}"] = df_lag["y"].shift(lag)
    df_lag["rolling_mean_4"] = df_lag["y"].rolling(4).mean()
    return df_lag.dropna()

df_lagged = create_lag_features(df)
df_lagged = df_lagged[["ds", "y", "lag_1", "lag_2", "lag_3", "lag_4", "rolling_mean_4"]]

train_size = int(len(df_lagged) * 0.85)
train, test = df_lagged[:train_size], df_lagged[train_size:]
X_train, y_train = train.drop(["ds", "y"], axis=1), train["y"]
X_test, y_test = test.drop(["ds", "y"], axis=1), test["y"]

# ---- STEP 3: Train Model ----
model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
r2 = r2_score(y_test, preds)

print(f"✅ Training complete | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.3f}")

# ---- STEP 4: Log to MLflow ----
client = MlflowClient()
print(f"✅ Experiments visible: {[exp.name for exp in client.search_experiments()]}")

with mlflow.start_run(run_name=f"CI-CD-XGBoost-{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
    mlflow.log_params({
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9
    })
    mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})

    model_path = "best_model_xgb.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    print(f"🧾 MLflow Run ID: {run.info.run_id}")

print("🏁 Training pipeline finished successfully!")
