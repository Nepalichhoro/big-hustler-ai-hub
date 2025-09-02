import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

d_in = 4
d_model = 8
n_heads = 2
d_head = d_model // n_heads
d_ff = 16  # hidden size in FFN

# Define layers
W_q = nn.Linear(d_in, d_model)
W_k = nn.Linear(d_in, d_model)
W_v = nn.Linear(d_in, d_model)
W_o = nn.Linear(d_model, d_model)

ff1 = nn.Linear(d_model, d_ff)
ff2 = nn.Linear(d_ff, d_model)

norm1 = nn.LayerNorm(d_model)
norm2 = nn.LayerNorm(d_model)

# Input: 2 tokens, 4 features each
x = torch.tensor([
    [
        [20000.0, 20010.0, 19990.0, 100.0],
        [15000.0, 15010.0, 14990.0, 200.0]
    ]
], dtype=torch.float32)

# ---- Self-Attention ----
q = W_q(x)
k = W_k(x)
v = W_v(x)

q_heads = q.view(1, 2, n_heads, d_head).transpose(1, 2)
k_heads = k.view(1, 2, n_heads, d_head).transpose(1, 2)
v_heads = v.view(1, 2, n_heads, d_head).transpose(1, 2)

scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (d_head ** 0.5)
attn_weights = F.softmax(scores, dim=-1)
attn_output = torch.matmul(attn_weights, v_heads)

concat = attn_output.transpose(1, 2).contiguous().view(1, 2, d_model)
attn_out = W_o(concat)

# ---- Residual + Norm 1 ----
residual1 = x.new_zeros(attn_out.shape)  # if input dim != d_model, we’d add projection
residual1[:, :, :d_in] = x  # crude shortcut to align shapes
normed1 = norm1(attn_out + residual1)

# ---- FeedForward ----
ff_out = ff2(F.relu(ff1(normed1)))

# ---- Residual + Norm 2 ----
out = norm2(normed1 + ff_out)

print("\n=== Shape Trace (Encoder Block) ===")
print(f"Input              : {x.shape}")
print(f"Q/K/V              : {q.shape}")
print(f"Heads              : {q_heads.shape}")
print(f"Scores             : {scores.shape}")
print(f"Attention output   : {attn_output.shape}")
print(f"Concat             : {concat.shape}")
print(f"After W₀           : {attn_out.shape}")
print(f"Norm1              : {normed1.shape}")
print(f"FFN out            : {ff_out.shape}")
print(f"Final Encoder out  : {out.shape}")

print("\nFinal Encoder Output:\n", out)

'''

Input: (1, 2, 4)
   ↓ Linear W_q, W_k, W_v
Q, K, V: (1, 2, 8)
   ↓ reshape
Q/K/V heads: (1, 2, 2, 4)
   ↓ QKᵀ / √d_head
Attention scores: (1, 2, 2, 2)
   ↓ softmax × V
Attention output per head: (1, 2, 2, 4)
   ↓ concat
Concat heads: (1, 2, 8)
   ↓ Linear W₀
Attn out: (1, 2, 8)
   ↓ Residual + LayerNorm
Norm1: (1, 2, 8)
   ↓ FeedForward (Linear 8→16 → ReLU → Linear 16→8)
FFN out: (1, 2, 8)
   ↓ Residual + LayerNorm
Norm2 (Final Encoder Output): (1, 2, 8)


'''