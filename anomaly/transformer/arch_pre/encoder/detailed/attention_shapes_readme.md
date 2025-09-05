# Multi-Head Attention Shape Walkthrough

This document explains the shapes step-by-step for a simple example.

---

## Setup

- **Input shape**: `(1, 2, 4)` → batch size = 1, sequence length = 2,
  feature dimension = 4\
  (2 tokens, each with 4 features)
- **Model dimension**: `d_model = 8`
- **Number of heads**: `n_heads = 2`
- **Head dimension**: `d_head = d_model / n_heads = 4`

---

## Step-by-Step Flow

### 1. Input

    x: (1, 2, 4)

- Batch of 1
- Sequence length 2
- Feature size 4

### 2. Linear Projections (W_q, W_k, W_v)

Each input projected into model dimension = 8.

    Q, K, V: (1, 2, 8)

### 3. Reshape for Multi-Heads

Split into `n_heads = 2`, each of size `d_head = 4`.

    Q, K, V heads: (1, 2, 2, 4)
    # shape: (batch, seq_len, n_heads, d_head)

### 4. Compute Attention Scores

For each head, compute `Q @ Kᵀ / √d_head`.

    Scores: (1, 2, 2, 2)
    # For each head, scores are (seq_len, seq_len)

### 5. Apply Softmax (Row-wise Normalization)

    Weights: (1, 2, 2, 2)
    # Same shape as scores

### 6. Weighted Sum with V

    Attn output per head: (1, 2, 2, 4)

### 7. Concatenate Heads

    Concat: (1, 2, 8)

### 8. Final Linear Projection (W₀)

    Output: (1, 2, 8)

---

## Summary Table

---

Step Shape Notes

---

Input (1, 2, 4) 2 tokens, 4
features
each

Q, K, V (linear proj) (1, 2, 8) project to
d_model

Split into heads (1, 2, 2, 4) 2 heads × 4
dim each

Attention scores (QKᵀ) (1, 2, 2, 2) per head
similarity

Softmax weights (1, 2, 2, 2) normalized
scores

Weighted sum with V (1, 2, 2, 4) per-head
outputs

Concatenate heads (1, 2, 8) join head
outputs

Final linear W₀ (1, 2, 8) model
dimension
result

---

---

**Key idea:** Scores and weights always have shape
`(seq_len, seq_len)` per head. Softmax does not change size, only
normalizes each row into probabilities.
