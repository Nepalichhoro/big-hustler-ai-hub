from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from typing import List

# ----------------------------
# Data Preprocessing
# ----------------------------
def generate_mock_btc(n_points=300, seed=42):
    """Generate mock BTC dataset with value, max, min, count columns."""
    np.random.seed(seed)
    value = np.cumsum(np.random.randn(n_points)) + 20000
    rolling_max = pd.Series(value).rolling(5, min_periods=1).max().values
    rolling_min = pd.Series(value).rolling(5, min_periods=1).min().values
    rolling_count = np.arange(1, n_points+1)

    # Fake anomaly labels: random 5% anomalies
    labels = np.zeros(n_points)
    anomalies_idx = np.random.choice(n_points, size=int(0.05*n_points), replace=False)
    labels[anomalies_idx] = 1

    df = pd.DataFrame({
        "value": value,
        "max": rolling_max,
        "min": rolling_min,
        "count": rolling_count,
        "label": labels
    })
    return df

def scale_data(df):
    scaler = MinMaxScaler()
    features = df[["value", "max", "min", "count"]]
    data_scaled = scaler.fit_transform(features)
    return data_scaled, df["label"].values, scaler

# ----------------------------
# Model Definition (XGBoost)
# ----------------------------
def train_xgb(data_scaled, labels):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(data_scaled, labels)
    return model

def compute_xgb_anomalies(model, data_scaled):
    probs = model.predict_proba(data_scaled)[:, 1]  # probability of being anomaly
    anomalies = np.where(probs > 0.5)[0].tolist()
    return probs.tolist(), 0.5, anomalies

# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI(title="BTC XGBoost Anomaly Detection API")

df = generate_mock_btc()
data_scaled, labels, scaler = scale_data(df)
model = train_xgb(data_scaled, labels)

# ----------------------------
# Request / Response Models
# ----------------------------
class BTCDataPoint(BaseModel):
    value: float
    max: float
    min: float
    count: int

class PredictionResponse(BaseModel):
    anomalies: List[int]
    threshold: float
    errors: List[float]

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/")
def root():
    return {"message": "BTC XGBoost Anomaly Detection API running 🚀"}

@app.get("/detect", response_model=PredictionResponse)
def detect_anomalies():
    errors, threshold, anomalies = compute_xgb_anomalies(model, data_scaled)
    return {
        "anomalies": anomalies,
        "threshold": threshold,
        "errors": errors
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: List[BTCDataPoint]):
    new_df = pd.DataFrame([d.dict() for d in data])
    new_scaled = scaler.transform(new_df)

    errors, threshold, anomalies = compute_xgb_anomalies(model, new_scaled)
    return {
        "anomalies": anomalies,
        "threshold": threshold,
        "errors": errors
    }

@app.post("/train")
def retrain(data: List[BTCDataPoint]):
    global model, scaler, data_scaled

    new_df = pd.DataFrame([d.dict() for d in data])
    new_df["label"] = np.random.choice([0, 1], size=len(new_df), p=[0.95, 0.05])  # fake labels

    data_scaled, labels, scaler = scale_data(new_df)
    model = train_xgb(data_scaled, labels)

    return {"message": "XGBoost model retrained successfully on new data."}
