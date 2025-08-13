import torch
import torch.nn as nn

# --------------------
# Hyperparameters
# --------------------
embedding_dim = 8
hidden_size = 16
output_size = 4
sequence_length = 5   # time steps
batch_size = 1        # easier to visualize

# --------------------
# Layers
# --------------------
Wx = nn.Linear(embedding_dim, hidden_size)
Wh = nn.Linear(hidden_size, hidden_size)
Wy = nn.Linear(hidden_size, output_size)

# --------------------
# Fake embeddings for sequence
# --------------------
# Shape: (sequence_length, embedding_dim)
sequence_embeddings = torch.randn(sequence_length, embedding_dim)

# --------------------
# Initial hidden state
# --------------------
h_prev = torch.zeros(hidden_size)

print(f"Initial hidden state (h_0):\n{h_prev}\n{'-'*50}")

# --------------------
# Loop over time steps
# --------------------
for t in range(sequence_length):
    x_t = sequence_embeddings[t]  # embedding vector for time step t

    # Input and context contributions
    input_contrib = Wx(x_t)         # from embedding
    context_contrib = Wh(h_prev)    # from previous hidden state

    # New hidden state
    h_t = torch.tanh(input_contrib + context_contrib)

    # Output
    y_t = Wy(h_t)

    print(f"Time step {t+1}")
    print(f"Embedding vector (x_t):\n{x_t}")
    print(f"Input contribution Wx(x_t):\n{input_contrib}")
    print(f"Context contribution Wh(h_prev):\n{context_contrib}")
    print(f"New hidden state (h_t):\n{h_t}")
    print(f"Output (y_t):\n{y_t}")
    print("-"*50)

    # Update hidden state for next step
    h_prev = h_t
