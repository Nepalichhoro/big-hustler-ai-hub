# From Raw Time Series to LSTM Batches

## Example Setup

- Raw data points: **1000**
- Sequence length (`seq_len`): **10**
- Batch size: **32**

---

## Step 1: Raw Data

We start with a univariate time series:

```
[x1, x2, x3, ..., x1000]
```

Shape: `(1000, 1)`

---

## Step 2: Sliding Windows

We create overlapping windows of length 10:

```
[x1,  x2,  x3,  ..., x10 ]
[x2,  x3,  x4,  ..., x11 ]
[x3,  x4,  x5,  ..., x12 ]
 ...
[x991, x992, ..., x1000]
```

- Total windows = `1000 - seq_len = 990`
- Each window shape = `(10, 1)`

So the **Dataset** has 990 samples.

---

## Step 3: DataLoader Batching

With `batch_size = 32`, the DataLoader groups windows together:

```
Batch 1:
[[x1..x10],
 [x2..x11],
 ...
 [x32..x41]]

Batch 2:
[[x33..x42],
 [x34..x43],
 ...
 [x64..x73]]

...
```

- Each batch shape = `(32, 10, 1)`
- Matches PyTorch LSTM expected input: `(batch_size, seq_len, input_size)`

---

## Diagram

```
Raw Series (1000 points)
 └── Sliding Windows (seq_len=10)
       └── 990 overlapping windows
             └── Grouped into batches of 32
                   └── Shape = (32, 10, 1)
```

---

## Summary

- **Raw data (1000 points)** → `(1000,1)`
- **Sliding windows (seq_len=10)** → `990` samples, each `(10,1)`
- **DataLoader batching (batch_size=32)** → `(32,10,1)` per batch

This is exactly the shape an LSTM needs when `batch_first=True`.
