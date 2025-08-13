embedding_dim = 8
hidden_size = 16
output_size = 4

# Layers
Wx = nn.Linear(embedding_dim, hidden_size)
Wh = nn.Linear(hidden_size, hidden_size)
Wy = nn.Linear(hidden_size, output_size)

# Input (embedding vector) and previous hidden state
x_t = torch.randn(embedding_dim)
h_prev = torch.zeros(hidden_size)

# Compute hidden state
h_t = torch.tanh(Wx(x_t) + Wh(h_prev))

# Compute output
y_t = Wy(h_t)

print("Embedding vector:", x_t.shape)
print("Hidden state:", h_t.shape)
print("Output:", y_t.shape)
