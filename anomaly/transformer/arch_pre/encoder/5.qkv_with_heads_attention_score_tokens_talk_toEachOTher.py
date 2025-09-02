import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)  # reproducibility

d_in = 4       # input feature size
d_model = 8    # embedding dim (projection size)
n_heads = 2
d_head = d_model // n_heads  # per-head dimension

# Define linear projections
W_q = nn.Linear(d_in, d_model)
W_k = nn.Linear(d_in, d_model)
W_v = nn.Linear(d_in, d_model)

# Example input: batch=1, seq_len=2, features=4
x = torch.tensor([
    [
        [20000.0, 20010.0, 19990.0, 100.0],   # token 1
        [15000.0, 15010.0, 14990.0, 200.0]    # token 2
    ]
], dtype=torch.float32)

# Compute Q, K, V (shape (1,2,8))
q = W_q(x)
k = W_k(x)
v = W_v(x)

# Reshape into heads: (batch, n_heads, seq_len, d_head)
q_heads = q.view(1, 2, n_heads, d_head).transpose(1, 2)
k_heads = k.view(1, 2, n_heads, d_head).transpose(1, 2)
v_heads = v.view(1, 2, n_heads, d_head).transpose(1, 2)

# Attention scores: QK^T / sqrt(d_head)
scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (d_head ** 0.5)
attn_weights = F.softmax(scores, dim=-1)

# Attention output: softmax * V
attn_output = torch.matmul(attn_weights, v_heads)

print("Input shape:", x.shape)
print("Input:", x)

print("\nQ split into heads:", q_heads.shape)
print("Q heads:\n", q_heads)

print("\nK split into heads:", k_heads.shape)
print("K heads:\n", k_heads)

print("\nV split into heads:", v_heads.shape)
print("V heads:\n", v_heads)

print("\nAttention scores (before softmax):", scores.shape)
print(scores)

print("\nAttention weights (after softmax):", attn_weights.shape)
print(attn_weights)

print("\nAttention output:", attn_output.shape)
print(attn_output)

'''

What you’ll see

scores now has shape (1, 2, 2, 2):

batch=1, n_heads=2, seq_len=2 (Q) × seq_len=2 (K).

So each head has a 2×2 matrix of raw attention scores.

Diagonal = self-attention (token attends to itself).

Off-diagonal = cross-attention (token attends to the other token).

attn_weights: after softmax along last dim, each row sums to 1.

Row 0 = how much token 1 attends to [token1, token2].

Row 1 = how much token 2 attends to [token1, token2].

attn_output: weighted combination of values V for each head.

'''