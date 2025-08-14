import torch
import torch.nn as nn

torch.manual_seed(42)

# Dimensions
embedding_dim = 8
hidden_size   = 16
output_size   = 4

# ---- Define weights as parameters ----
Wx = nn.Parameter(torch.randn(embedding_dim, hidden_size))   # input → hidden
Wh = nn.Parameter(torch.randn(hidden_size, hidden_size))     # hidden → hidden
Wy = nn.Parameter(torch.randn(hidden_size, output_size))     # hidden → output

# Biases
b_h = nn.Parameter(torch.zeros(hidden_size))
b_y = nn.Parameter(torch.zeros(output_size))

# ---- Input and previous hidden state ----
x_t = torch.randn(1, embedding_dim)   # (batch=1, emb_dim=8)
h_prev = torch.zeros(1, hidden_size)  # (batch=1, hidden_size=16)

print("x_t (embedding vector):\n", x_t)
print("\nWx shape:", Wx.shape)
print("Wh shape:", Wh.shape)
print("Wy shape:", Wy.shape)

# ---- Step 1: Hidden state update ----
# (1×8) @ (8×16)  → (1×16)
x_part = x_t @ Wx

# (1×16) @ (16×16) → (1×16)
h_part = h_prev @ Wh

# Sum and add bias
h_t = torch.tanh(x_part + h_part + b_h)
print("\nNew hidden state h_t:\n", h_t)

# ---- Step 2: Output ----
# (1×16) @ (16×4) → (1×4)
y_t = h_t @ Wy + b_y
print("\nOutput y_t:\n", y_t)

# ---- Shapes ----
print("\nShapes:")
print("Embedding vector:", x_t.shape)  # (1, 8)
print("Hidden state:", h_t.shape)      # (1, 16)
print("Output:", y_t.shape)            # (1, 4)
