import torch
import torch.nn as nn

torch.manual_seed(42)

# --------------------
# Hyperparameters
# --------------------
embedding_dim   = 8
hidden_size     = 16
output_size     = 4
sequence_length = 4   # time steps
batch_size      = 3   # 3 sequences in parallel

# --------------------
# Parameters (raw matrices & biases)
# --------------------
# Shapes chosen so we can use (batch, in) @ (in, out) -> (batch, out)
Wx = nn.Parameter(torch.randn(embedding_dim, hidden_size))   # input → hidden
Wh = nn.Parameter(torch.randn(hidden_size, hidden_size))     # hidden → hidden
Wy = nn.Parameter(torch.randn(hidden_size, output_size))     # hidden → output

b_h = nn.Parameter(torch.zeros(hidden_size))
b_y = nn.Parameter(torch.zeros(output_size))

# --------------------
# Fake embeddings for batch of sequences
# --------------------
# Shape: (batch_size, sequence_length, embedding_dim)
sequence_embeddings = torch.randn(batch_size, sequence_length, embedding_dim)

# --------------------
# Initial hidden state for all sequences in batch
# --------------------
h_prev = torch.zeros(batch_size, hidden_size)  # (3, 16)

print(f"Initial hidden state (h_0) shape: {h_prev.shape}\n{'-'*50}")

# --------------------
# Loop over time steps
# --------------------
for t in range(sequence_length):
    # x_t: (batch, emb_dim)
    x_t = sequence_embeddings[:, t, :]

    # Input and context contributions
    # (batch, 8)  @ (8,16)   -> (batch, 16)
    input_contrib   = x_t @ Wx
    # (batch, 16) @ (16,16)  -> (batch, 16)
    context_contrib = h_prev @ Wh

    # New hidden state: add bias b_h (broadcasts to (batch, 16))
    h_t = torch.tanh(input_contrib + context_contrib + b_h)

    # Output: (batch, 16) @ (16, 4) -> (batch, 4), then add b_y
    y_t = h_t @ Wy + b_y

    print(f"Time step {t+1}")
    print(f"Embedding vector batch shape: {x_t.shape}")             # (3, 8)
    print(f"Input contribution shape: {input_contrib.shape}")       # (3, 16)
    print(f"Context contribution shape: {context_contrib.shape}")   # (3, 16)
    print(f"New hidden state shape: {h_t.shape}")                   # (3, 16)
    print(f"Output shape: {y_t.shape}")                             # (3, 4)
    print("-"*50)

    # Update hidden state for next time step
    h_prev = h_t
