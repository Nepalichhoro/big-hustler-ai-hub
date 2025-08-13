import torch
import torch.nn as nn

# --------------------
# Hyperparameters
# --------------------
vocab_size = 10         # Number of unique tokens
embedding_dim = 8       # Embedding dimension (dense vector size per token)
hidden_size = 16        # Size of RNN hidden state (context vector size)
sequence_length = 5     # Number of tokens per sequence
batch_size = 2          # Number of sequences in a batch

# --------------------
# Dummy input
# --------------------
# Shape: (batch_size, sequence_length)
# Each number is a token ID from 0 to vocab_size-1
inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
print("Token IDs shape:", inputs.shape)  # (2, 5)

# --------------------
# Step 1: Embedding layer
# --------------------
embedding = nn.Embedding(vocab_size, embedding_dim)
embedded = embedding(inputs)
print("Embedded shape:", embedded.shape)  # (2, 5, 8)
# (batch_size, seq_length, embedding_dim)

# --------------------
# Step 2: Elman RNN
# --------------------
class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wx = nn.Linear(input_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)
        self.Wy = nn.Linear(hidden_size, output_size)
        self.activation = torch.tanh

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size)  # initial hidden state (context=0)
        outputs = []

        for t in range(seq_len):  # TIME STEPS loop
            x_t = x[:, t, :]       # (batch_size, embedding_dim)
            h = self.activation(self.Wx(x_t) + self.Wh(h))  # context from previous h
            y_t = self.Wy(h)       # output at time step t
            outputs.append(y_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (batch_size, seq_len, output_size)

# --------------------
# Step 3: Run model
# --------------------
model = ElmanRNN(embedding_dim, hidden_size, output_size=4)
outputs = model(embedded)

print("Output shape:", outputs.shape)  # (batch_size, seq_len, output_size)
