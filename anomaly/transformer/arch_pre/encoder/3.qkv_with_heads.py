import torch
import torch.nn as nn

torch.manual_seed(0)  # reproducibility

d_in = 4       # input feature size
d_model = 8    # embedding dim (projection size)
n_heads = 2
d_head = d_model // n_heads  # per-head dimension

# Define linear projections
W_q = nn.Linear(d_in, d_model)
W_k = nn.Linear(d_in, d_model)
W_v = nn.Linear(d_in, d_model)

# Example input: batch=1, seq_len=1, features=4
x = torch.tensor([[[20000.0, 20010.0, 19990.0, 100.0]]], dtype=torch.float32)

# Compute Q, K, V (still shape (1,1,8))
q = W_q(x)
k = W_k(x)
v = W_v(x)

# Reshape into heads: (batch, seq_len, n_heads, d_head)
q_heads = q.view(1, 1, n_heads, d_head)
k_heads = k.view(1, 1, n_heads, d_head)
v_heads = v.view(1, 1, n_heads, d_head)

print("Input shape:", x.shape)
print("Input:", x)

print("\nQ shape:", q.shape)
print("Q values:\n", q)

print("\nQ split into heads:", q_heads.shape)
print("Q heads:\n", q_heads)

print("\nK split into heads:", k_heads.shape)
print("K heads:\n", k_heads)

print("\nV split into heads:", v_heads.shape)
print("V heads:\n", v_heads)
