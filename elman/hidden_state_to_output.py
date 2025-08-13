import torch
import torch.nn as nn

hidden_size = 16
output_size = 4  # Example: maybe predicting 4 categories

# Random new hidden state from previous computation
h_t = torch.randn(hidden_size)

# Hidden -> Output layer
Wy = nn.Linear(hidden_size, output_size)

# Compute output
y_t = Wy(h_t)

print("Hidden state shape:", h_t.shape)   # (16,)
print("Output shape:", y_t.shape)         # (4,)
print(y_t)
