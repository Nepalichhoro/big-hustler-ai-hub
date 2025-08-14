import torch
import torch.nn as nn

torch.manual_seed(42)

hidden_size = 16
output_size = 4  # Example: predicting 4 categories

# --------------------
# Random hidden state from previous computation
# --------------------
h_t = torch.randn(hidden_size)  # shape: (16,)

# --------------------
# Wy as a raw parameter
# Shape: (hidden_size, output_size) so (16, 4)
# --------------------
Wy = nn.Parameter(torch.randn(hidden_size, output_size))
b_y = nn.Parameter(torch.zeros(output_size))

# --------------------
# Compute output
# (16,) @ (16, 4) -> (4,)
# --------------------
y_t = h_t @ Wy + b_y

print("Hidden state shape:", h_t.shape)   # torch.Size([16])
print("Wy shape:", Wy.shape)              # torch.Size([16, 4])
print("Output shape:", y_t.shape)         # torch.Size([4])
print("\nOutput vector:\n", y_t)
