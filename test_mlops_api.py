# test_mlops_api.py — API test
import requests

BASE_URL = "http://localhost:8000"

def test_health():
    r = requests.get(f"{BASE_URL}/")
    print("Health:", r.json())

def test_predict():
    sample_input = {
        "lag_1": 450.5,
        "lag_2": 430.2,
        "lag_3": 400.0,
        "lag_4": 380.5,
        "rolling_mean_4": 415.0
    }
    r = requests.post(f"{BASE_URL}/predict", json=sample_input)
    print("Prediction:", r.json())

if __name__ == "__main__":
    print("🔍 Checking API health...")
    test_health()
    print("\n📈 Sending prediction request...")
    test_predict()
