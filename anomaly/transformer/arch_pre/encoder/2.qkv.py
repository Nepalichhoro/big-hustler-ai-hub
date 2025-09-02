import torch
import torch.nn as nn

torch.manual_seed(0)  # reproducibility

d_in = 4      # input feature size
d_model = 8   # embedding dim (projection size)

# Define linear projections
W_q = nn.Linear(d_in, d_model)
W_k = nn.Linear(d_in, d_model)
W_v = nn.Linear(d_in, d_model)

# Example input: batch=1, seq_len=1, features=4
x = torch.tensor([[[20000.0, 20010.0, 19990.0, 100.0]]])  

# Compute Q, K, V
q = W_q(x)   # (1,1,8)
k = W_k(x)   # (1,1,8)
v = W_v(x)   # (1,1,8)

print("Input shape:", x.shape)
print("Input:", x)

print("\nQ shape:", q.shape)
print("Q values:\n", q)

print("\nK shape:", k.shape)
print("K values:\n", k)

print("\nV shape:", v.shape)
print("V values:\n", v)
