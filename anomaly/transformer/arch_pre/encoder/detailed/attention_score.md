
# 🧩 Transformer Encoder (Mini Version) — README

This README shows a **step-by-step attention calculation** with:
- Input: `(1, 2, 4)` → 2 tokens, each 4 features  
- Model dimension: `d_model = 8`  
- Number of heads: `n_heads = 2`  
- Head dimension: `d_head = 4`  

We’ll walk through: **Q/K/V projection → split heads → scores → weights → weighted sum → concat → output projection → residuals + norms → FFN.**

---

## 🔢 Step-by-Step Flow

### 1. Input
```
Input: (1, 2, 4)
[[[1., 2., 3., 4.],
  [5., 6., 7., 8.]]]
```

---

### 2. Linear Projections
```
↓ Linear W_q, W_k, W_v
Q, K, V: (1, 2, 8)
[[[1., 2., 3., 4., 0., 0., 0., 0.],
  [5., 6., 7., 8., 0., 0., 0., 0.]]]
```

---

### 3. Split into Heads
```
↓ reshape
Q/K/V heads: (1, 2, 2, 4)

Head0:
[[1., 2., 3., 4.],
 [5., 6., 7., 8.]]

Head1:
[[0., 0., 0., 0.],
 [0., 0., 0., 0.]]
```

---

### 4. Attention Scores
```
↓ QKᵀ / √d_head
Head0 Scores:
[[15., 35.],
 [35., 87.]]

Head1 Scores:
[[0., 0.],
 [0., 0.]]
```

---

### 5. Attention Weights (Softmax)
```
Head0 Weights:
[[~0.0, 1.0],
 [~0.0, 1.0]]

Head1 Weights:
[[0.5, 0.5],
 [0.5, 0.5]]
```

---

### 6. Multiply by V (Weighted Sum)
```
Head0 Output:
[[5., 6., 7., 8.],
 [5., 6., 7., 8.]]

Head1 Output:
[[0., 0., 0., 0.],
 [0., 0., 0., 0.]]
```

---

### 7. Concatenate Heads
```
Concat heads: (1, 2, 8)
[[5., 6., 7., 8., 0., 0., 0., 0.],
 [5., 6., 7., 8., 0., 0., 0., 0.]]
```

---

### 8. Apply W₀ (Output Projection)
```
↓ Linear W₀
Attn out: (1, 2, 8)
(same as concat when W₀ = Identity)
```

---

### 9. Residual + LayerNorm
```
↓ Residual + Norm1
[[ 0.3145,  0.7338,  1.1531,  1.5724, -0.9435, -0.9435, -0.9435, -0.9435],
 [ 0.5232,  0.8222,  1.1212,  1.4201, -0.9717, -0.9717, -0.9717, -0.9717]]
```

---

### 10. FeedForward (FFN)
```
↓ Linear 8→16 → ReLU → Linear 16→8
[[-0.1593, 0.2188, -0.1640, -0.1524, -0.2531, -0.2986, 0.3109, 0.0736],
 [-0.1071, 0.2309, -0.1382, -0.1433, -0.2563, -0.2987, 0.2668, 0.0719]]
```

---

### 11. Residual + LayerNorm
```
↓ Residual + Norm2 (Final Output)
[[ 0.2077,  1.0034,  1.0399,  1.4698, -1.1411, -1.1864, -0.5782, -0.8150],
 [ 0.4547,  1.0805,  1.0115,  1.3002, -1.1604, -1.2020, -0.6465, -0.8380]]
```

---

# 🔁 Manual Loop Implementation (Both Heads + Concat)

```python
import math

def attention_loop(Q, K, V):
    seq_len, d_k = len(Q), len(Q[0])
    scores = [[0]*seq_len for _ in range(seq_len)]
    for i in range(seq_len):  # query token
        for j in range(seq_len):  # key token
            scores[i][j] = sum(Q[i][k]*K[j][k] for k in range(d_k)) / math.sqrt(d_k)
    # softmax row-wise
    weights = []
    for i in range(seq_len):
        row = scores[i]
        exps = [math.exp(x) for x in row]
        s = sum(exps)
        weights.append([e/s for e in exps])
    # weighted sum of V
    outputs = []
    for i in range(seq_len):
        out = [0.0]*d_k
        for j in range(seq_len):
            for k in range(d_k):
                out[k] += weights[i][j]*V[j][k]
        outputs.append(out)
    return scores, weights, outputs

# Head0 example
Q0 = [[1,2,3,4],[5,6,7,8]]
K0, V0 = Q0, Q0
scores0, weights0, out0 = attention_loop(Q0, K0, V0)

# Head1 example (zeros for simplicity)
Q1 = [[0,0,0,0],[0,0,0,0]]
K1, V1 = Q1, Q1
scores1, weights1, out1 = attention_loop(Q1, K1, V1)

# Concatenate heads
concat = [out0[i] + out1[i] for i in range(len(out0))]
print("Concatenated output:", concat)
```

---

# ⚡ PyTorch Vectorized Version

```python
import torch, torch.nn as nn, torch.nn.functional as F
import math

B, T, d_in, d_model, n_heads = 1, 2, 4, 8, 2
d_head = d_model // n_heads

x = torch.arange(1.0, B*T*d_in+1).view(B, T, d_in)

W_q, W_k, W_v = nn.Linear(d_in, d_model, bias=False), nn.Linear(d_in, d_model, bias=False), nn.Linear(d_in, d_model, bias=False)
Q, K, V = W_q(x), W_k(x), W_v(x)

def split_heads(t):
    return t.view(B, T, n_heads, d_head).transpose(1,2)

Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

scores = torch.matmul(Qh, Kh.transpose(-2,-1)) / math.sqrt(d_head)
weights = F.softmax(scores, dim=-1)
attn_out = torch.matmul(weights, Vh)
concat = attn_out.transpose(1,2).contiguous().view(B, T, d_model)

W_o = nn.Linear(d_model, d_model)
attn_proj = W_o(concat)
```

---
