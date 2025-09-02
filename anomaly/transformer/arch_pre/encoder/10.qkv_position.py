import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)

d_in = 4
d_model = 8
n_heads = 2
d_head = d_model // n_heads
d_ff = 16  # hidden size in FFN

# ---- Positional Encoding (sinusoidal) ----
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (seq_len, d_model)

# ---- Layers ----
input_proj = nn.Linear(d_in, d_model)  # project input → d_model
W_q = nn.Linear(d_model, d_model)
W_k = nn.Linear(d_model, d_model)
W_v = nn.Linear(d_model, d_model)
W_o = nn.Linear(d_model, d_model)

ff1 = nn.Linear(d_model, d_ff)
ff2 = nn.Linear(d_ff, d_model)

norm1 = nn.LayerNorm(d_model)
norm2 = nn.LayerNorm(d_model)

# ---- Input: 2 tokens, 4 features each ----
x = torch.tensor([
    [
        [20000.0, 20010.0, 19990.0, 100.0],   # token 1
        [15000.0, 15010.0, 14990.0, 200.0]    # token 2
    ]
], dtype=torch.float32)  # (1, 2, 4)

# ---- Step 1: Project + Add Positional Encoding ----
seq_len = x.size(1)
x_proj = input_proj(x)                               # (1, 2, 8)
pe = positional_encoding(seq_len, d_model).unsqueeze(0)  # (1, 2, 8)
x_with_pos = x_proj + pe                             # (1, 2, 8)

# ---- Step 2: Self-Attention ----
q = W_q(x_with_pos)
k = W_k(x_with_pos)
v = W_v(x_with_pos)

q_heads = q.view(1, seq_len, n_heads, d_head).transpose(1, 2)
k_heads = k.view(1, seq_len, n_heads, d_head).transpose(1, 2)
v_heads = v.view(1, seq_len, n_heads, d_head).transpose(1, 2)

scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (d_head ** 0.5)
attn_weights = F.softmax(scores, dim=-1)
attn_output = torch.matmul(attn_weights, v_heads)

concat = attn_output.transpose(1, 2).contiguous().view(1, seq_len, d_model)
attn_out = W_o(concat)

# ---- Step 3: Residual + Norm 1 ----
normed1 = norm1(attn_out + x_with_pos)

# ---- Step 4: FeedForward ----
ff_out = ff2(F.relu(ff1(normed1)))

# ---- Step 5: Residual + Norm 2 ----
out = norm2(normed1 + ff_out)

# ---- Shape Trace ----
print("\n=== Shape Trace (Encoder Block with Positional Encoding) ===")
print(f"Input raw          : {x.shape}")
print(f"Input projected    : {x_proj.shape}")
print(f"Positional enc.    : {pe.shape}")
print(f"Input+pos          : {x_with_pos.shape}")
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
