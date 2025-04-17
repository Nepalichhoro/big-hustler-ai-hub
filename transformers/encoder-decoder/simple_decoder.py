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
        return x + self.pe[:, :x.size(1)]


class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(T, T), diagonal=1).to(torch.bool).to(x.device)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        output = attn @ v
        concat = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(concat)


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.multihead_attn(query=x, key=encoder_output, value=encoder_output)
        return attn_output


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048):
        super().__init__()
        self.self_attn = MaskedMultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = CrossAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.self_attn(x))
        x = self.norm2(x + self.cross_attn(x, encoder_output))
        x = self.norm3(x + self.ffn(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        max_len: int = 512,
        d_ff: int = 2048
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tgt)
        x = self.pos_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output)

        return self.output_proj(x)  # logits

vocab_size = 1000
d_model = 512
num_heads = 8
num_layers = 6
batch_size = 2
tgt_seq_len = 10
src_seq_len = 12

tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
encoder_output = torch.randn(batch_size, src_seq_len, d_model)

decoder = TransformerDecoder(vocab_size, d_model, num_layers, num_heads)
output = decoder(tgt, encoder_output)

print("Decoder Output Shape:", output.shape)  # [B, tgt_seq_len, vocab_size]
