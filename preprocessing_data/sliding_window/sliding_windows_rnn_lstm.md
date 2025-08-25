# Sliding Windows for RNN/LSTM Time Series Models

## Why Sliding Windows?

Recurrent models like **RNNs** and **LSTMs** don’t process the entire raw time series at once.  
Instead, they learn from **fixed-length subsequences** of the series.

Because real time series are often very long (thousands or millions of points), we create **sliding windows**:

- Each window is a contiguous slice of the series.
- Windows overlap so the model sees every local temporal context.
- Each window has the same length (`seq_len`), making input shapes consistent for the model.

This is the **standard way to feed temporal data** into recurrent networks.

---

## Example of Sliding Windows

Suppose we have a univariate series:

```text
Data = [1, 2, 3, 4, 5, 6, 7, 8]
```

If `seq_len = 3`, sliding windows are:

```text
[1, 2, 3]
[2, 3, 4]
[3, 4, 5]
[4, 5, 6]
[5, 6, 7]
[6, 7, 8]
```

- Each row is one training sample.
- If the model is an autoencoder, the **target output = input window**.
- If the model is for forecasting, the **target could be the next value(s)** after the window.

---

## Tensor Shapes for LSTMs

PyTorch LSTMs expect input in the shape:

```text
(batch_size, seq_len, input_size)
```

- `batch_size`: number of windows per batch
- `seq_len`: length of each window (number of time steps)
- `input_size`: number of features per time step
  - `1` for univariate series
  - `>1` for multivariate (multiple signals at each timestamp)

Example:  
If we have 1000 data points, `seq_len = 10`, and `batch_size = 32`:

- Dataset produces ~990 windows of shape `(10, 1)`
- DataLoader groups them into batches of `(32, 10, 1)`

---

## PyTorch Dataset Implementation

A reusable class for sliding windows:

```python
import torch
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, data, seq_len):
        """
        Args:
            data (array-like): 1D numpy array or list of time series values
            seq_len (int): Length of each window
        """
        if isinstance(data, list):
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)
        elif isinstance(data, torch.Tensor):
            data = data.float().unsqueeze(-1) if data.ndim == 1 else data.float()
        else:  # assume numpy
            data = torch.from_numpy(data).float().unsqueeze(-1)

        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len]  # shape: (seq_len, 1)
```

Usage:

```python
from torch.utils.data import DataLoader

series = [1,2,3,4,5,6,7,8]
dataset = SlidingWindowDataset(series, seq_len=3)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch in loader:
    print(batch.shape)  # (batch_size, seq_len, 1)
```

Output:

```text
torch.Size([2, 3, 1])
torch.Size([2, 3, 1])
torch.Size([2, 3, 1])
```

---

## Where It Fits in Training

1. **Data preparation**  
   Normalize values (e.g., `MinMaxScaler`) → sliding windows.

2. **Dataset + DataLoader**  
   Create `SlidingWindowDataset` → wrap in `DataLoader`.

3. **Model input**  
   Feed `(batch_size, seq_len, input_size)` batches to the LSTM.

4. **Loss**

   - Autoencoder: compare reconstructed window vs input window.
   - Forecasting: compare predicted next values vs ground truth.

5. **Anomaly detection (autoencoder)**  
   Compute reconstruction error per window → flag anomalies above threshold.

---

## Key Tips

- **Window length matters**

  - Too short → model misses long-term patterns.
  - Too long → harder to train, more memory.

- **Stride**

  - Default is stride=1 (maximal overlap).
  - You can stride >1 to reduce dataset size.

- **Multivariate**
  - Instead of `(seq_len, 1)`, use `(seq_len, num_features)`.

---

## Summary

Sliding windows transform raw time series into fixed-length, overlapping subsequences.  
These subsequences become the standard training units for LSTMs/RNNs, enabling models to learn temporal dependencies and patterns effectively.
