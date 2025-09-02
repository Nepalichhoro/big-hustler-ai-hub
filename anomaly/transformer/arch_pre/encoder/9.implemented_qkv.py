import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)  # reproducibility

class DebugEncoderLayer(nn.Module):
    def __init__(self, d_model=4, nhead=2, dim_feedforward=8):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead  # size per head

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection after concatenating heads
        self.W_o = nn.Linear(d_model, d_model)

        # Feedforward network
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        print("\n=== Forward Pass ===")
        print("Input:", x.shape, "\n", x)

        B, T, _ = x.shape  # batch, seq_len, d_model

        # ---- 1. Compute Q, K, V ----
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        print("\nQ:", Q.shape, "\n", Q)
        print("K:", K.shape, "\n", K)
        print("V:", V.shape, "\n", V)

        # ---- 2. Split into heads ----
        Q = Q.view(B, T, self.nhead, self.d_head).transpose(1, 2)  # (B, nhead, T, d_head)
        K = K.view(B, T, self.nhead, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.nhead, self.d_head).transpose(1, 2)
        print("\nQ split into heads:", Q.shape)

        # ---- 3. Attention scores ----
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, nhead, T, T)
        attn_weights = F.softmax(scores, dim=-1)
        print("\nAttention scores (before softmax):", scores)
        print("\nAttention weights (after softmax):", attn_weights)

        # ---- 4. Attention output ----
        attn_out = torch.matmul(attn_weights, V)  # (B, nhead, T, d_head)
        print("\nAttention output per head:", attn_out.shape, "\n", attn_out)

        # ---- 5. Concatenate heads ----
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        attn_out = self.W_o(attn_out)
        print("\nAfter concatenating heads:", attn_out.shape, "\n", attn_out)

        # ---- 6. Residual + Norm ----
        x = self.norm1(x + attn_out)
        print("\nAfter residual+norm1:", x.shape, "\n", x)

        # ---- 7. Feedforward ----
        ff = F.relu(self.ff1(x))
        ff = self.ff2(ff)
        print("\nFeedforward output:", ff.shape, "\n", ff)

        # ---- 8. Residual + Norm ----
        out = self.norm2(x + ff)
        print("\nFinal output:", out.shape, "\n", out)

        return out
