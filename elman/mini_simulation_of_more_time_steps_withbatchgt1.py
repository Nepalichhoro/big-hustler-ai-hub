import torch
import torch.nn as nn

# --------------------
# Hyperparameters
# --------------------
embedding_dim = 8
hidden_size = 16
output_size = 4
sequence_length = 4   # time steps
batch_size = 3        # now processing 3 sequences in parallel

# --------------------
# Layers
# --------------------
Wx = nn.Linear(embedding_dim, hidden_size)
Wh = nn.Linear(hidden_size, hidden_size)
Wy = nn.Linear(hidden_size, output_size)

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
    x_t = sequence_embeddings[:, t, :]  # (batch_size, embedding_dim)

    # Input and context contributions
    input_contrib = Wx(x_t)         # (batch_size, hidden_size)
    context_contrib = Wh(h_prev)    # (batch_size, hidden_size)

    # New hidden state
    h_t = torch.tanh(input_contrib + context_contrib)  # (batch_size, hidden_size)

    # Output
    y_t = Wy(h_t)  # (batch_size, output_size)

    print(f"Time step {t+1}")
    print(f"Embedding vector batch shape: {x_t.shape}")        # (3, 8)
    print(f"Input contribution shape: {input_contrib.shape}")  # (3, 16)
    print(f"Context contribution shape: {context_contrib.shape}")  # (3, 16)
    print(f"New hidden state shape: {h_t.shape}")               # (3, 16)
    print(f"Output shape: {y_t.shape}")                         # (3, 4)
    print("-"*50)

    # Update hidden state for next time step
    h_prev = h_t
