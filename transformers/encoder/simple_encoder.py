import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # shape: [B, T, 3 * C]
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # [B, heads, T, d_k]

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, heads, T, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v  # [B, heads, T, d_k]

        concat = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, d_model]
        return self.out_proj(concat)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 4: Multi-Head Self-Attention
        attn_output = self.attn(x)

        # Step 5: Residual + LayerNorm
        x = self.norm1(x + attn_output)

        # Step 6: Feed Forward
        ffn_output = self.ffn(x)

        # Step 7: Residual + LayerNorm
        x = self.norm2(x + ffn_output)

        # Step 8: Output of Encoder Block
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, max_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 & 2: Word Embedding + Positional Encoding
        x = self.embedding(x)  # [B, seq_len, d_model]
        x = self.pos_encoding(x)  # [B, seq_len, d_model]

        # Step 3–8: Encoder Layers
        for layer in self.encoder_layers:
            x = layer(x)

        return x  # Final encoder output


# Mock input
batch_size = 2
seq_len = 10
vocab_size = 1000
d_model = 512
num_heads = 8
num_layers = 6

x = torch.randint(0, vocab_size, (batch_size, seq_len))  # token indices

encoder = TransformerEncoder(vocab_size, d_model, num_layers, num_heads)
output = encoder(x)  # [batch_size, seq_len, d_model]
