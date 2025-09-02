from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
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

class SequenceDataset(Dataset):
    def __init__(self, data, seq_len=10):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        return torch.tensor(x, dtype=torch.float32)

def scale_data(df):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    return data_scaled, scaler


# ----------------------------
# Model Definition
# ----------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, n_features, batch_first=True)
    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h_repeated = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        decoded, _ = self.decoder(h_repeated)
        return decoded


# ----------------------------
# Training & Evaluation Utils
# ----------------------------
def create_dataloader(data_scaled, seq_len=10, batch_size=16):
    dataset = SequenceDataset(data_scaled, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset

def train_model(model, dataloader, n_epochs=3, lr=0.01, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(n_epochs):
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def compute_errors(model, dataset, device="cpu"):
    criterion = nn.MSELoss()
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x = dataset[i].unsqueeze(0).to(device)
            output = model(x)
            loss = criterion(output, x)
            errors.append(loss.item())
    errors = np.array(errors)
    threshold = np.mean(errors) + 2*np.std(errors)
    anomalies = np.where(errors > threshold)[0].tolist()
    return errors.tolist(), threshold, anomalies


# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI(title="BTC LSTM Anomaly Detection API")

# Global objects (init with mock data)
device = "cpu"
df = generate_mock_btc()
data_scaled, scaler = scale_data(df)
dataloader, dataset = create_dataloader(data_scaled, seq_len=10)
model = LSTMAutoencoder(n_features=4, hidden_dim=8)
model = train_model(model, dataloader, n_epochs=5, device=device)


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
    return {"message": "BTC LSTM Anomaly Detection API running 🚀"}

@app.get("/detect", response_model=PredictionResponse)
def detect_anomalies():
    """Run anomaly detection on the baseline mock dataset (pretrained at startup)."""
    errors, threshold, anomalies = compute_errors(model, dataset, device=device)
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
    new_dataset = SequenceDataset(new_scaled, seq_len=10)

    errors, threshold, anomalies = compute_errors(model, new_dataset, device=device)
    return {
        "anomalies": anomalies,
        "threshold": threshold,
        "errors": errors
    }

@app.post("/train")
def retrain(data: List[BTCDataPoint]):
    """Retrain the model on new dataset provided by user."""
    global model, scaler, dataset, dataloader

    new_df = pd.DataFrame([d.dict() for d in data])
    new_scaled, scaler = scale_data(new_df)
    dataloader, dataset = create_dataloader(new_scaled, seq_len=10)

    model = LSTMAutoencoder(n_features=4, hidden_dim=8)
    model = train_model(model, dataloader, n_epochs=5, device=device)

    return {"message": "Model retrained successfully on new data."}
