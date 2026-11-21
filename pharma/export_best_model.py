# ===============================================
# 📦 Export Best Model from MLflow – Pharma Project
# ===============================================
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
import joblib

# Must match your experiment name from the notebook
EXPERIMENT_NAME = "Pharma Weekly Forecast (Tuned)"  

def main(out_path="models/best_model_xgb.pkl"):
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)

    if exp is None:
        print(f"❌ ERROR: Experiment '{EXPERIMENT_NAME}' not found.")
        print("➡️  Run your notebook first to create MLflow runs.")
        sys.exit(1)

    # Search runs sorted by highest R² (best performing)
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.R2 DESC"]
    )

    if runs.empty:
        print("⚠️ No runs found. Train models before exporting.")
        sys.exit(1)

    # Pick top run
    best = runs.iloc[0]
    run_id = best["run_id"]
    best_r2 = best.get("metrics.R2", None)
    print(f"🏆 Best run_id = {run_id}  (R² = {best_r2:.4f})")

    # Construct MLflow model URI
    model_uri = f"runs:/{run_id}/model"
    print(f"📦 Loading model from MLflow URI: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print("❌ Failed to load model from MLflow:", e)
        sys.exit(1)

    # Ensure models folder exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)
    print(f"✅ Saved best model to {out_path}")

if __name__ == "__main__":
    main()
