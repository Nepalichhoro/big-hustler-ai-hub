import torch
import torch.nn as nn

class DebugTransformer(nn.Module):
    def __init__(self, n_features=1, d_model=8, nhead=2):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # (1) Project input feature -> d_model
        self.input_projection = nn.Linear(n_features, d_model)

        # (2) Positional embeddings (trainable for demo)
        self.pos_embedding = nn.Parameter(torch.randn(1, 50, d_model))

        # (3) Multi-head attention
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

    def forward(self, x):
        print("\n=== Forward Pass ===")
        print("Raw input:", x.shape)  # (B, T, 1)

        # (1) Projection
        x_proj = self.input_projection(x)  # (B, T, d_model)
        print("After projection:", x_proj.shape)

        # (2) Add positional encoding
        seq_len = x.size(1)
        x_proj = x_proj + self.pos_embedding[:, :seq_len, :]
        print("After positional encoding:", x_proj.shape)

        # (3) Multi-head attention
        attn_out, attn_weights = self.mha(x_proj, x_proj, x_proj, need_weights=True)
        print("After attention output:", attn_out.shape)       # (B, T, d_model)
        print("Attention weights:", attn_weights.shape)        # (B, heads, T, T)

        # Log one attention matrix
        print("\nSample attention matrix (batch 0, head 0):")
        print(attn_weights[0, 0].detach().cpu().numpy())

        return attn_out

# Fake BTC values: batch=1, seq_len=5, features=1
x = torch.tensor([[[20000.0], [20010.0], [19990.0], [20050.0], [20030.0]]])
# print("Input values:\n", x.squeeze(-1).numpy())

model = DebugTransformer(n_features=1, d_model=8, nhead=2)
# out = model(x)

# print("\nFinal Output:", out.shape)
