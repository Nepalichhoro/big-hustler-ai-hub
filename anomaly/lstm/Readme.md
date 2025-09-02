# /detect →

Runs anomaly detection on the mock BTC dataset (the one generated and trained when the FastAPI app starts).

Use it to see how the model behaves on baseline BTC data.

No input required, always runs on the preloaded dataset.

Think of it as a built-in “demo mode.”

# /predict →

Accepts custom BTC data (with features: value, max, min, count) from the request body.

Scales the data with the same scaler used during training.

Creates a temporary dataset, runs inference, and returns anomalies.

Useful if you want to test real BTC prices or your own dataset without retraining.

# /train →

Accepts a new dataset (JSON list of BTC points) and retrains the model.

Reinitializes the LSTM autoencoder.

Updates scaler + dataset + model.

After this, both /detect and /predict will use the newly trained model.

Useful if you want to “reset” training with new data.
