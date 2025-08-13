import torch
import torch.nn as nn

hidden_size = 16

# Previous hidden state (context vector from last time step)
h_prev = torch.randn(hidden_size)  # shape: (16,)

# Linear transform: hidden_size -> hidden_size
Wh = nn.Linear(hidden_size, hidden_size)

# Transform the previous hidden state
context_contribution = Wh(h_prev)

print("Previous hidden state shape:", h_prev.shape)         # (16,)
print("Context contribution shape:", context_contribution.shape)  # (16,)
print(context_contribution)
