import torch
import torch.nn as nn

torch.manual_seed(42)

# --------------------
# Hyperparameters
# --------------------
embedding_dim = 8
hidden_size   = 16
output_size   = 4
sequence_length = 5   # time steps
batch_size      = 1  # visual clarity (we'll keep vectors 1D)

# --------------------
# Parameters (raw matrices & biases)
# --------------------
# Shapes: (in, out) to use x @ W (vector-matrix) style
Wx = nn.Parameter(torch.randn(embedding_dim, hidden_size))   # input → hidden
Wh = nn.Parameter(torch.randn(hidden_size, hidden_size))     # hidden → hidden
Wy = nn.Parameter(torch.randn(hidden_size, output_size))     # hidden → output

b_h = nn.Parameter(torch.zeros(hidden_size))
b_y = nn.Parameter(torch.zeros(output_size))

# --------------------
# Fake embeddings for sequence
# --------------------
# Shape: (sequence_length, embedding_dim)
sequence_embeddings = torch.randn(sequence_length, embedding_dim)

# --------------------
# Initial hidden state
# --------------------
h_prev = torch.zeros(hidden_size)  # (16,)

print(f"Initial hidden state (h_0):\n{h_prev}\n{'-'*50}")

# --------------------
# Loop over time steps
# --------------------
for t in range(sequence_length):
    x_t = sequence_embeddings[t]  # (8,)

    # Input and context contributions
    # x_t @ Wx: (8,)  @ (8,16)  -> (16,)
    # h_prev @ Wh: (16,) @ (16,16) -> (16,)
    input_contrib   = x_t @ Wx
    context_contrib = h_prev @ Wh

    # New hidden state
    h_t = torch.tanh(input_contrib + context_contrib + b_h)

    # Output
    # h_t @ Wy: (16,) @ (16,4) -> (4,)
    y_t = h_t @ Wy + b_y

    print(f"Time step {t+1}")
    print(f"Embedding vector (x_t) shape: {tuple(x_t.shape)}")
    print(f"Embedding vector (x_t):\n{x_t}")
    print(f"\nInput contribution x_t @ Wx (shape {tuple(input_contrib.shape)}):\n{input_contrib}")
    print(f"\nContext contribution h_prev @ Wh (shape {tuple(context_contrib.shape)}):\n{context_contrib}")
    print(f"\nNew hidden state (h_t) (shape {tuple(h_t.shape)}):\n{h_t}")
    print(f"\nOutput (y_t) (shape {tuple(y_t.shape)}):\n{y_t}")
    print("-"*50)

    # Update hidden state for next step
    h_prev = h_t
