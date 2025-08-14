import torch
import torch.nn as nn

# -----------------------------
# 1) One Elman RNN unit (x_t, h_{t-1}) -> h_t
# -----------------------------
class ElmanRNNUnit(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        # Weight for previous hidden state (U) and for input (W)
        self.Uh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.Wh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        # Bias
        self.b  = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x, h):
        # x: (batch, emb_dim), h: (batch, emb_dim)
        # @ is matrix multiply; broadcasting adds b
        return torch.tanh(x @ self.Wh + h @ self.Uh + self.b)


# -----------------------------
# 2) Multi-layer Elman RNN built from ElmanRNNUnit
# -----------------------------
class ElmanRNN(nn.Module):
    def __init__(self, emb_dim: int, num_layers: int):
        super().__init__()
        self.emb_dim    = emb_dim
        self.num_layers = num_layers
        self.rnn_units  = nn.ModuleList([ElmanRNNUnit(emb_dim) for _ in range(num_layers)])

    def forward(self, x):
        """
        x: (batch_size, seq_len, emb_dim)
        returns:
          outputs stacked over time: (batch_size, seq_len, emb_dim)
          (i.e., the output of the top layer at each time step)
        """
        batch_size, seq_len, emb_dim = x.shape

        # One hidden state tensor per layer; init to zeros
        h_prev = [torch.zeros(batch_size, emb_dim, device=x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            input_t = x[:, t]  # (batch, emb_dim)
            for l, rnn_unit in enumerate(self.rnn_units):
                h_new       = rnn_unit(input_t, h_prev[l])  # compute new hidden
                h_prev[l]   = h_new                         # update in place
                input_t     = h_new                         # pass up to next layer
            # after last layer, collect output for this time step
            outputs.append(input_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, emb_dim)


# -----------------------------
# 3) Tiny demo (remove if not needed)
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, emb_dim, num_layers = 2, 5, 8, 2
    x = torch.randn(batch_size, seq_len, emb_dim)  # pretend these are embeddings

    rnn = ElmanRNN(emb_dim=emb_dim, num_layers=num_layers)
    y = rnn(x)
    print("Input shape:", x.shape)   # (2, 5, 8)
    print("Output shape:", y.shape)  # (2, 5, 8)  ← top-layer outputs across time
