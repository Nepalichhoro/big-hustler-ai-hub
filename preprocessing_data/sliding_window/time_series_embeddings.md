# Embedding-Style Enrichment for Time Series

## Introduction

In **NLP**, embeddings are dense vector representations of tokens that capture semantic meaning (e.g., Word2Vec, BERT).  
For **time series**, embeddings serve a similar purpose: enrich raw values or timestamps with extra representational power for downstream models like RNNs, LSTMs, or Transformers.

---

## Raw Time Series vs Embeddings

- **Raw input (univariate series)**

  ```
  [1.2, 3.5, 2.8, 4.1, 5.0]
  ```

  Each value is just a scalar (1D feature).

- **Sliding window**

  ```
  [1.2, 3.5, 2.8] → one training sample
  ```

  Defines _context_ but does not enrich features.

- **Embedding-style enrichment**  
  Each time step is mapped into a richer representation vector (like tokens in NLP):
  ```
  1.2 → [0.8, -0.2, 0.5, ...]
  3.5 → [0.3,  0.7, 0.1, ...]
  ```

Now, instead of feeding plain scalars, the model gets **feature-rich vectors**.

---

## Techniques for Time Series Embeddings

### 1. **Learned Value Embeddings**

- Treat discrete values (e.g., event IDs, categories, sensor IDs) as tokens.
- Use `nn.Embedding` in PyTorch to map each category to a dense vector.
- Useful for **multivariate categorical series**.

```python
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=50, embedding_dim=16)
```

---

### 2. **Time Feature Embeddings**

Enrich each time step with features derived from the timestamp:

- Hour of day (0–23)
- Day of week (0–6)
- Month of year (1–12)

These categorical values can also be embedded.  
Example: “Wednesday at 3pm in March” becomes a vector, not just a scalar.

---

### 3. **Positional / Temporal Encoding**

Borrowed from Transformers:

- Add a **positional encoding** vector to each time step to capture its place in the sequence.
- Or use **sinusoidal features** like Fourier terms to represent seasonality.

```math
PE(t,2i)   = sin(t / 10000^(2i/d))
PE(t,2i+1) = cos(t / 10000^(2i/d))
```

---

### 4. **Time2Vec**

A learnable embedding for time proposed by Kazemi et al. (NeurIPS 2019).

- Represents time as a combination of **linear** and **periodic** components.
- Captures seasonality, cycles, and trends in a compact vector form.

```math
f(t)[0]   = ω₀ t + φ₀
f(t)[i]   = sin(ωᵢ t + φᵢ),   i = 1..k
```

---

### 5. **Lag Features & Domain-Specific Embeddings**

- Add embeddings for lagged values (e.g., x(t-1), x(t-7), x(t-30))
- Use Fourier transforms, wavelets, or learned filters to embed frequency information.
- Domain knowledge: encode calendar effects, holidays, business hours, etc.

---

## Analogy with NLP

| NLP (Text)                      | Time Series (Signals)                   |
| ------------------------------- | --------------------------------------- |
| Token embedding (Word2Vec/BERT) | Value embedding (categorical sensors)   |
| Positional encoding             | Temporal encoding (time-of-day, season) |
| Sentence context (n-grams)      | Sliding windows (seq_len samples)       |
| Learned semantic space          | Enriched feature space (Time2Vec, lags) |

---

## Summary

- **Sliding windows** give structure but not enrichment.
- **Embedding-style techniques** (categorical embeddings, time feature embeddings, Time2Vec, positional encodings) **turn raw time steps into rich vectors**.
- This allows models like RNNs, LSTMs, and Transformers to capture **seasonality, trends, and categorical context** far better than raw scalars alone.

Think of embeddings as **teaching time series models to understand “semantics of time”**, just like word embeddings let NLP models understand language meaning.
