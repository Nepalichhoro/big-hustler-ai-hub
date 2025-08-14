'''
see how an 8-dim embedding vector becomes a 16-dim hidden contribution via Wx
'''

import torch
import torch.nn as nn

# Embedding vector (example from you)
embedding_vec = torch.tensor([
    0.7936,  1.2883,  0.0174,  0.7390,
   -1.2885, -0.6574,  1.6326,  1.4787
])

embedding_dim = 8
hidden_size = 16

# Linear layer: input_size=8 → hidden_size=16
Wx = nn.Linear(embedding_dim, hidden_size)

# Pass the embedding through Wx
hidden_contribution = Wx(embedding_vec)

print("Embedding vector shape:", embedding_vec.shape)       # (8,)
print("Hidden contribution shape:", hidden_contribution.shape)  # (16,)
print(hidden_contribution)
