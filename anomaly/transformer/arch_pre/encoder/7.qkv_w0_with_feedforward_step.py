import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

d_in = 4
d_model = 8
n_heads = 2
d_head = d_model // n_heads
d_ff = 16  # feedforward hidden size

# Define projections + FFN
W_q = nn.Linear(d_in, d_model)
W_k = nn.Linear(d_in, d_model)
W_v = nn.Linear(d_in, d_model)
W_o = nn.Linear(d_model, d_model)

ff1 = nn.Linear(d_model, d_ff)
ff2 = nn.Linear(d_ff, d_model)

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
out = W_o(concat)

# ---- Feedforward ----
ff_out = ff2(F.relu(ff1(out)))

print("\n=== Shape Trace ===")
print(f"Input              : {x.shape}")
print(f"Q, K, V            : {q.shape}")
print(f"Q/K/V split heads  : {q_heads.shape}")
print(f"Attention scores   : {scores.shape}")
print(f"Attention output   : {attn_output.shape}")
print(f"Concat heads       : {concat.shape}")
print(f"After W₀           : {out.shape}")
print(f"Feedforward output : {ff_out.shape}")

print("\nFinal Encoder Output:\n", ff_out)


'''
Input: (1, 2, 4)            # batch=1, seq_len=2, d_in=4
   ↓
Linear W_q, W_k, W_v
   ↓
Q, K, V: (1, 2, 8)          # project to d_model=8
   ↓ reshape
Split into heads: (1, 2, 2, 4)   # n_heads=2, d_head=4
   ↓
Attention scores (QKᵀ / √d_head): (1, 2, 2, 2)
   ↓ softmax
Attention weights × V
   ↓
Attn output (per head): (1, 2, 2, 4)
   ↓ concat
Concat heads: (1, 2, 8)
   ↓ Linear W₀
Output after W₀: (1, 2, 8)
   ↓
Feedforward Layer:
   Linear (8 → 16) → ReLU → Linear (16 → 8)
   ↓
Final Encoder Output: (1, 2, 8)
'''
