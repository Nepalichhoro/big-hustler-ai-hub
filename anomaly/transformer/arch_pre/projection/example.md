import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# Parameters

batch = 1
seq_len = 2 # two tokens
d_in = 4 # input features per token
d_model = 9 # projection size
n_heads = 3
d_head = d_model // n_heads # = 3

# Projection layers

W_q = nn.Linear(d_in, d_model, bias=False)
W_k = nn.Linear(d_in, d_model, bias=False)
W_v = nn.Linear(d_in, d_model, bias=False)
W_o = nn.Linear(d_model, d_model, bias=False)

# Input: 2 tokens, each with 4 features

x = torch.randn(batch, seq_len, d_in)
print("Input:", x.shape)

# ---- Linear projections ----

q = W_q(x) # (1, 2, 9)
k = W_k(x) # (1, 2, 9)
v = W_v(x) # (1, 2, 9)
print("\nAfter projections: Q/K/V:", q.shape)

# ---- Split into heads ----

q_heads = q.view(batch, seq_len, n_heads, d_head).transpose(1, 2) # (1, 3, 2, 3)
k_heads = k.view(batch, seq_len, n_heads, d_head).transpose(1, 2)
v_heads = v.view(batch, seq_len, n_heads, d_head).transpose(1, 2)
print("Q_heads:", q_heads.shape)

# ---- Attention ----

scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (d_head \*\* 0.5)
attn_weights = F.softmax(scores, dim=-1)
attn_output = torch.matmul(attn_weights, v_heads)
print("Scores:", scores.shape) # (1, 3, 2, 2) ← square matrices per head
print("Attn Output per head:", attn_output.shape)

# ---- Concat ----

concat = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
print("Concat:", concat.shape) # (1, 2, 9)

# ---- Final output ----

out = W_o(concat)
print("Final Output:", out.shape) # (1, 2, 9)
