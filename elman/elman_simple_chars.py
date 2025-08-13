import torch
import torch.nn as nn

# --------------------
# Example vocabulary
# --------------------
vocab = ["i", "love", "cats", "you", "hate", "dogs", "and", "birds", "eat", "fish"]
word_to_id = {word: idx for idx, word in enumerate(vocab)}

# Two sequences (padded to same length = 5)
seq1 = ["i", "love", "cats", "and", "dogs"]
seq2 = ["you", "hate", "birds", "and", "fish"]

# Convert words to token IDs
inputs_list = [
    [word_to_id[w] for w in seq1],
    [word_to_id[w] for w in seq2]
]

# Shape: (batch_size, sequence_length)
inputs = torch.tensor(inputs_list)
print("Token IDs:\n", inputs)

# --------------------
# Hyperparameters
# --------------------
vocab_size = len(vocab) # 10
embedding_dim = 8
hidden_size = 16
output_size = 4  # just an arbitrary output vector size
sequence_length = inputs.shape[1]
batch_size = inputs.shape[0]

# --------------------
# Step 1: Embedding
# --------------------
embedding = nn.Embedding(vocab_size, embedding_dim)
embedded = embedding(inputs)
print("\nEmbedded shape:", embedded.shape)  # (2, 5, 8)

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
        h = torch.zeros(batch_size, self.hidden_size)  # initial hidden state
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]       # embedding for token at time t
            h = self.activation(self.Wx(x_t) + self.Wh(h))
            y_t = self.Wy(h)
            outputs.append(y_t.unsqueeze(1))

            print(f"\nTime step {t+1}:")
            print("  Token IDs:", inputs[:, t])
            print("  Words:    ", [vocab[idx] for idx in inputs[:, t]])
            print("  h_t shape:", h.shape)
            print("  y_t shape:", y_t.shape)

        return torch.cat(outputs, dim=1)

# --------------------
# Step 3: Run model
# --------------------
model = ElmanRNN(embedding_dim, hidden_size, output_size)
outputs = model(embedded)

print("\nFinal output shape:", outputs.shape)  # (2, 5, 4)
print("Final outputs:\n", outputs)
