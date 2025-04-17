# Transformer Encoder Block: End-to-End Summary

| Step | Layer / Component                | Input Shape        | Operation                                     | Output Shape       | Type of Operation            |
| ---- | -------------------------------- | ------------------ | --------------------------------------------- | ------------------ | ---------------------------- |
| 1    | Word Embedding                   | [seq_len]          | Map token to vector via embedding             | [seq_len, d_model] | Lookup (learned embedding)   |
| 2    | Positional Encoding              | [seq_len, d_model] | Sinusoidal or learned positional vector       | [seq_len, d_model] | Precomputed/static           |
| 3    | Add Word + Positional            | [seq_len, d_model] | Element-wise: w + p                           | [seq_len, d_model] | Element-wise addition        |
| 4    | Multi-Head Self-Attention        | [seq_len, d_model] | Compute Q, K, V, attention weights            | [seq_len, d_model] | Scaled dot-product attention |
| 5    | Residual + LayerNorm (Post-Attn) | [seq_len, d_model] | input + attention_output                      | [seq_len, d_model] | Element-wise addition + norm |
| 6    | Position-wise Feed Forward       | [seq_len, d_model] | 2-layer MLP per token: Linear → ReLU → Linear | [seq_len, d_model] | Token-wise (1x1 conv-like)   |
| 7    | Residual + LayerNorm (Post-FFN)  | [seq_len, d_model] | input + ffn_output                            | [seq_len, d_model] | Element-wise addition + norm |
| 8    | Output of Encoder Block          | [seq_len, d_model] | Passed to next encoder or decoder             | [seq_len, d_model] | —                            |
