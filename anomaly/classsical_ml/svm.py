from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from typing import List

# ----------------------------
# Data Preprocessing
# ----------------------------
def generate_mock_btc(n_points=300, seed=42):
    """Generate mock BTC dataset with value, max, min, count columns."""
    np.random.seed(seed)
    value = np.cumsum(np.random.randn(n_points)) + 20000  # simulate price
    rolling_max = pd.Series(value).rolling(5, min_periods=1).max().values
    rolling_min = pd.Series(value).rolling(5, min_periods=1).min().values
    rolling_count = np.arange(1, n_points+1)

    df = pd.DataFrame({
        "value": value,
        "max": rolling_max,
        "min": rolling_min,
        "count": rolling_count
    })
    return df

def scale_data(df):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    return data_scaled, scaler

# ----------------------------
# Model Definition (SVM)
# ----------------------------
def train_svm(data_scaled, nu=0.05, kernel="rbf", gamma="scale"):
    """Train One-Class SVM for anomaly detection."""
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(data_scaled)
    return model

def compute_svm_anomalies(model, data_scaled):
    """Compute anomaly scores and detect anomalies."""
    preds = model.predict(data_scaled)  # +1 normal, -1 anomaly
    scores = model.decision_function(data_scaled)  # anomaly scores (higher = more normal)
    anomalies = np.where(preds == -1)[0].tolist()
    threshold = np.min(scores)  # decision boundary reference
    return scores.tolist(), threshold, anomalies

# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI(title="BTC SVM Anomaly Detection API")

# Global objects (init with mock data)
df = generate_mock_btc()
data_scaled, scaler = scale_data(df)
model = train_svm(data_scaled)

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
    return {"message": "BTC SVM Anomaly Detection API running 🚀"}

@app.get("/detect", response_model=PredictionResponse)
def detect_anomalies():
    """Run anomaly detection on the baseline mock dataset (pretrained at startup)."""
    errors, threshold, anomalies = compute_svm_anomalies(model, data_scaled)
    return {
        "anomalies": anomalies,
        "threshold": threshold,
        "errors": errors
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: List[BTCDataPoint]):
    """Run anomaly detection on custom BTC dataset without retraining."""
    new_df = pd.DataFrame([d.dict() for d in data])
    new_scaled = scaler.transform(new_df)

    errors, threshold, anomalies = compute_svm_anomalies(model, new_scaled)
    return {
        "anomalies": anomalies,
        "threshold": threshold,
        "errors": errors
    }

@app.post("/train")
def retrain(data: List[BTCDataPoint]):
    """Retrain the SVM model on new dataset provided by user."""
    global model, scaler, data_scaled

    new_df = pd.DataFrame([d.dict() for d in data])
    data_scaled, scaler = scale_data(new_df)

    model = train_svm(data_scaled)

    return {"message": "SVM model retrained successfully on new data."}
