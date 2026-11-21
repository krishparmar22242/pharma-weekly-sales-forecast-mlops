# ============================================================
# 📈 Flask API for Pharma Weekly Sales Forecast (MLOps Ready)
# ============================================================
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


# --- Model Path ---
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model_xgb.pkl")

# --- Load model ---
model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded best model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

# --- Utility: Create lag features (same as training) ---
def create_lag_features(df, lags=[1, 2, 3, 4]):
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['rolling_mean_4'] = df['y'].rolling(4).mean()
    df = df.dropna()
    return df

# --- Home route ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "📊 Pharma Weekly Forecast API is running!",
        "usage": "POST /predict with recent weekly sales data."
    })

# --- Predict route ---
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
        "recent_sales": [410.5, 420.7, 389.3, 450.8],
        "weeks_ahead": 8   # optional (default=4)
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    data = request.get_json(force=True)
    if not data or "recent_sales" not in data:
        return jsonify({
            "error": "Missing 'recent_sales' list in JSON body."
        }), 400

    recent_sales = data["recent_sales"]
    weeks_ahead = int(data.get("weeks_ahead", 4))

    if len(recent_sales) < 4:
        return jsonify({
            "error": "At least 4 recent weekly sales values are required."
        }), 400

    try:
        # Prepare dataframe for feature creation
        df = pd.DataFrame({"y": recent_sales})
        df_lagged = create_lag_features(df)

        # Extract last row for current lags
        latest_features = df_lagged.iloc[-1].drop("y").values.reshape(1, -1)

        # Rolling predictions
        preds = []
        current_features = latest_features.copy()

        for _ in range(weeks_ahead):
            pred = model.predict(current_features)[0]
            preds.append(round(float(pred), 2))

            # Update lag window manually
            recent_sales.append(pred)
            temp_df = pd.DataFrame({"y": recent_sales[-5:]})
            temp_lagged = create_lag_features(temp_df)
            current_features = temp_lagged.iloc[-1].drop("y").values.reshape(1, -1)

        response = {
            "weeks_requested": weeks_ahead,
            "predicted_sales": preds
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
