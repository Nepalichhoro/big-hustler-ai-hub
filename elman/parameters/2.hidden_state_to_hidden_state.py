import torch
import torch.nn as nn

torch.manual_seed(42)

hidden_size = 16

# --------------------
# Previous hidden state (context vector from last time step)
# --------------------
h_prev = torch.randn(hidden_size)  # shape: (16,)

# --------------------
# Wh as a raw parameter
# Shape: (hidden_size, hidden_size) so (16, 16)
# --------------------
Wh = nn.Parameter(torch.randn(hidden_size, hidden_size))
b_h = nn.Parameter(torch.zeros(hidden_size))

# --------------------
# Transform the previous hidden state
# (16,) @ (16, 16) -> (16,)
# --------------------
context_contribution = h_prev @ Wh + b_h

print("Previous hidden state shape:", h_prev.shape)              # torch.Size([16])
print("Wh shape:", Wh.shape)                                     # torch.Size([16, 16])
print("Context contribution shape:", context_contribution.shape) # torch.Size([16])
print("\nContext contribution vector:\n", context_contribution)
