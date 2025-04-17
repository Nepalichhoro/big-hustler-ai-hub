# Transformer Decoder Block: End-to-End Summary

| Step | Layer / Component                      | Input Shape                                              | Operation                                          | Output Shape           | Type of Operation                    |
| ---- | -------------------------------------- | -------------------------------------------------------- | -------------------------------------------------- | ---------------------- | ------------------------------------ |
| 1    | Word Embedding                         | [tgt_seq_len]                                            | Map target token to embedding vector               | [tgt_seq_len, d_model] | Lookup (learned embedding)           |
| 2    | Positional Encoding                    | [tgt_seq_len, d_model]                                   | Sinusoidal or learned                              | [tgt_seq_len, d_model] | Precomputed/static                   |
| 3    | Add Word + Positional                  | [tgt_seq_len, d_model]                                   | Element-wise: w + p                                | [tgt_seq_len, d_model] | Element-wise addition                |
| 4    | Masked Multi-Head Self-Attention       | [tgt_seq_len, d_model]                                   | QKV projection + causal mask (no future lookahead) | [tgt_seq_len, d_model] | Contextualized token embedding       |
| 5    | Residual + LayerNorm (Post Self-Attn)  | [tgt_seq_len, d_model]                                   | input + self_attn_output                           | [tgt_seq_len, d_model] | Element-wise addition + norm         |
| 6    | Encoder-Decoder Cross-Attention        | Q: [tgt_seq_len, d_model]<br>K,V: [src_seq_len, d_model] | Decoder attends to encoder output                  | [tgt_seq_len, d_model] | Cross-sequence attention             |
| 7    | Residual + LayerNorm (Post Cross-Attn) | [tgt_seq_len, d_model]                                   | input + cross_attn_output                          | [tgt_seq_len, d_model] | Element-wise addition + norm         |
| 8    | Position-wise Feed Forward             | [tgt_seq_len, d_model]                                   | 2-layer MLP: Linear → ReLU → Linear                | [tgt_seq_len, d_model] | Token-wise transformation (1x1 conv) |
| 9    | Residual + LayerNorm (Post FFN)        | [tgt_seq_len, d_model]                                   | input + ffn_output                                 | [tgt_seq_len, d_model] | Element-wise addition + norm         |
| 10   | Decoder Block Output                   | [tgt_seq_len, d_model]                                   | Passed to next decoder block or generator          | [tgt_seq_len, d_model] | —                                    |
