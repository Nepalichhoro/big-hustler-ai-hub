# Comparison Table: d_model and Key Specs

| Model           | Params     | d_model (hidden size) | Layers | Heads | Context Length | Notes                                     |
| --------------- | ---------- | --------------------- | ------ | ----- | -------------- | ----------------------------------------- |
| GPT-3 (Ada)     | 350M       | 768                   | 24     | 12    | 2048           | Smallest GPT-3 model                      |
| GPT-3 (Curie)   | 6.7B       | 4096                  | 32     | 32    | 2048           | Mid-size                                  |
| GPT-3 (Davinci) | 175B       | 12288                 | 96     | 96    | 2048           | Largest GPT-3                             |
| GPT-4           | ~1T (est.) | 12800–32000? (est.)   | ?      | ?     | 128K (Turbo)   | Exact specs unknown; significantly deeper |
| LLaMA 1 (7B)    | 7B         | 4096                  | 32     | 32    | 2048           | Meta's original LLaMA                     |
| LLaMA 2 (7B)    | 7B         | 4096                  | 32     | 32    | 4096           | Optimized and open weights                |
| LLaMA 2 (13B)   | 13B        | 5120                  | 40     | 40    | 4096           | Larger d_model                            |
| LLaMA 2 (70B)   | 70B        | 8192                  | 80     | 64    | 4096           | Deeper and wider                          |
| DeepSeek (7B)   | 7B         | 4096                  | 32     | 32    | 16K            | Open-source ChatGPT alternative           |
| DeepSeek (67B)  | 67B        | 8192                  | 80     | 64    | 32K            | High performance                          |
