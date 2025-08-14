import torch
import torch.nn as nn

torch.manual_seed(42)

# --------------------
# Embedding vector (example from you)
# --------------------
embedding_vec = torch.tensor([
    0.7936,  1.2883,  0.0174,  0.7390,
   -1.2885, -0.6574,  1.6326,  1.4787
])

embedding_dim = 8
hidden_size   = 16

# --------------------
# Wx as a raw parameter
# Shape: (embedding_dim, hidden_size) so (8, 16)
# --------------------
Wx = nn.Parameter(torch.randn(embedding_dim, hidden_size))
b_h = nn.Parameter(torch.zeros(hidden_size))

# --------------------
# Compute the hidden contribution
# --------------------
# (8,) @ (8, 16) -> (16,)
hidden_contribution = embedding_vec @ Wx + b_h

print("Embedding vector shape:", embedding_vec.shape)             # torch.Size([8])
print("Wx shape:", Wx.shape)                                      # torch.Size([8, 16])
print("Hidden contribution shape:", hidden_contribution.shape)    # torch.Size([16])
print("\nHidden contribution vector:\n", hidden_contribution)
