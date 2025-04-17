import torch
import torch.nn as nn
import math


# ============ MODEL COMPONENTS ============

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
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = attn_weights @ v
        concat = attn_output.transpose(1, 2).contiguous().view(B, T, C)
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
        attn_output = self.attn(x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
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
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x


# ============ TESTS ============

def test_positional_encoding():
    d_model = 16
    seq_len = 5
    batch_size = 2
    pos_enc = PositionalEncoding(d_model)
    x = torch.zeros(batch_size, seq_len, d_model)
    out = pos_enc(x)
    assert out.shape == (batch_size, seq_len, d_model)
    print("[✓] PositionalEncoding:", out.shape)

def test_multihead_self_attention():
    d_model = 32
    num_heads = 4
    seq_len = 6
    batch_size = 2
    x = torch.randn(batch_size, seq_len, d_model)
    attn = MultiHeadSelfAttention(d_model, num_heads)
    out = attn(x)
    assert out.shape == (batch_size, seq_len, d_model)
    print("[✓] MultiHeadSelfAttention:", out.shape)

def test_feedforward():
    d_model = 32
    d_ff = 64
    seq_len = 6
    batch_size = 2
    x = torch.randn(batch_size, seq_len, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff)
    out = ff(x)
    assert out.shape == (batch_size, seq_len, d_model)
    print("[✓] FeedForward:", out.shape)

def test_encoder_block():
    d_model = 64
    num_heads = 8
    d_ff = 128
    seq_len = 6
    batch_size = 2
    x = torch.randn(batch_size, seq_len, d_model)
    block = TransformerEncoderBlock(d_model, num_heads, d_ff)
    out = block(x)
    assert out.shape == (batch_size, seq_len, d_model)
    print("[✓] TransformerEncoderBlock:", out.shape)

def test_transformer_encoder():
    vocab_size = 1000
    d_model = 128
    num_heads = 8
    num_layers = 4
    seq_len = 10
    batch_size = 2
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    encoder = TransformerEncoder(vocab_size, d_model, num_layers, num_heads)
    out = encoder(x)
    assert out.shape == (batch_size, seq_len, d_model)
    print("[✓] TransformerEncoder:", out.shape)


# ============ RUN ALL TESTS ============

if __name__ == "__main__":
    test_positional_encoding()
    test_multihead_self_attention()
    test_feedforward()
    test_encoder_block()
    test_transformer_encoder()
