from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

# =====================================================
# Data Preprocessing + Feature Engineering
# =====================================================
def generate_mock_btc(n_points=300, seed=42):
    """Generate mock BTC dataset with value, max, min, count columns."""
    np.random.seed(seed)
    value = np.cumsum(np.random.randn(n_points)) + 20000  # simulate BTC-like price
    rolling_max = pd.Series(value).rolling(5, min_periods=1).max().values
    rolling_min = pd.Series(value).rolling(5, min_periods=1).min().values
    rolling_count = np.arange(1, n_points + 1)

    df = pd.DataFrame({
        "value": value,
        "max": rolling_max,
        "min": rolling_min,
        "count": rolling_count
    })
    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add some basic engineered features."""
    df = df.copy()
    # Returns (percentage change)
    df["return"] = df["value"].pct_change().fillna(0)
    # Volatility (rolling std dev of returns)
    df["volatility"] = df["return"].rolling(5, min_periods=1).std().fillna(0)
    # Spread (max - min)
    df["spread"] = df["max"] - df["min"]
    # Ratio (value relative to rolling max)
    df["value_ratio"] = df["value"] / (df["max"] + 1e-6)

    return df

def scale_data(df: pd.DataFrame):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    return data_scaled, scaler

# =====================================================
# Isolation Forest Anomaly Detector
# =====================================================
class BTCAnomalyDetector:
    def __init__(self, contamination=0.05, random_state=42):
        self.model = IsolationForest(contamination=contamination,
                                     random_state=random_state)
        self.scaler = None
        self.features = None

    def fit(self, df: pd.DataFrame):
        df_fe = feature_engineer(df)
        self.features = df_fe.columns.tolist()
        X, self.scaler = scale_data(df_fe)
        self.model.fit(X)

    def detect(self, df: pd.DataFrame):
        df_fe = feature_engineer(df)
        X = self.scaler.transform(df_fe[self.features])
        preds = self.model.predict(X)  # -1 = anomaly, 1 = normal
        scores = -self.model.decision_function(X)  # higher = more anomalous
        anomalies = np.where(preds == -1)[0].tolist()
        return preds.tolist(), scores.tolist(), anomalies

# =====================================================
# FastAPI Setup
# =====================================================
app = FastAPI(title="BTC Isolation Forest Anomaly Detection API")

# Initialize global model
df = generate_mock_btc()
detector = BTCAnomalyDetector(contamination=0.05)
detector.fit(df)

# =====================================================
# Request / Response Models
# =====================================================
class BTCDataPoint(BaseModel):
    value: float
    max: float
    min: float
    count: int

class PredictionResponse(BaseModel):
    anomalies: List[int]
    scores: List[float]
    predictions: List[int]

# =====================================================
# API Endpoints
# =====================================================
@app.get("/")
def root():
    return {"message": "BTC Isolation Forest Anomaly Detection API running 🚀"}

@app.get("/detect", response_model=PredictionResponse)
def detect_anomalies():
    """Detect anomalies on baseline mock dataset."""
    preds, scores, anomalies = detector.detect(df)
    return {
        "anomalies": anomalies,
        "predictions": preds,
        "scores": scores
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: List[BTCDataPoint]):
    """Run anomaly detection on custom BTC dataset without retraining."""
    new_df = pd.DataFrame([d.dict() for d in data])
    preds, scores, anomalies = detector.detect(new_df)
    return {
        "anomalies": anomalies,
        "predictions": preds,
        "scores": scores
    }

@app.post("/train")
def retrain(data: List[BTCDataPoint]):
    """Retrain Isolation Forest on new dataset."""
    global detector, df
    new_df = pd.DataFrame([d.dict() for d in data])
    df = new_df
    detector = BTCAnomalyDetector(contamination=0.05)
    detector.fit(new_df)
    return {"message": "Isolation Forest retrained successfully on new data."}
