# Understanding Projection in Attention

## What is Projection?

In linear algebra, a **projection** is just a linear transformation: multiplying a vector by a matrix.

You start with a vector in one space (say 4-D), and map it into another space (say 6-D).

Example:

```
q = x × W_q
```

If `x = (1, 2, 3, 4)` and `W_q` is a `4×6` matrix, you get a new vector `q` of length 6.

---

## Why Do We Project in Attention?

Each token embedding starts as a vector of size `d_in` (the number of input features).  
But attention needs three different roles: **query, key, and value**.

- **Query projection (`W_q`)**: “What am I looking for?”
- **Key projection (`W_k`)**: “What do I contain?”
- **Value projection (`W_v`)**: “What information do I pass if I’m attended to?”

So we apply three different projections to the same token:

```
q = x @ W_q
k = x @ W_k
v = x @ W_v
```

Each is a linear projection into a new space of dimension `d_model`.

---

## Projection vs. Slicing

Projection is **not** just slicing the embedding vector.  
If we only sliced the embedding vector, each head would always see the same fixed features.

By projecting, the model can **learn useful combinations of features** that make attention work better.

Example:

- Raw embedding (d_in = 4): `[color, shape, size, position]`
- After `W_q`: `[is_round, is_big, location_hint, ...]`
- After `W_k`: `[has_edges, is_small, left/right bias, ...]`

Now attention has **different optimized views** of the same token.

---

## Summary

- **Projection** = apply a learnable linear transformation (matrix multiply).
- This changes the representation of each token into a new space (`d_model`).
- Separate projections (`W_q`, `W_k`, `W_v`) provide different roles for queries, keys, and values.
- After projection, these vectors are split into heads and processed by multi-head attention.

---

## Toy Example in PyTorch

```python
import torch
import torch.nn as nn

# One token with 4 features
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # shape (1, 4)

# Projection: d_in=4 -> d_model=6
W_q = nn.Linear(4, 6, bias=False)
torch.manual_seed(0)
nn.init.xavier_uniform_(W_q.weight)

# Apply projection
q = W_q(x)
print("Input x:", x)
print("Projected q:", q)
```

Output (example):

```
Input x: [[1., 2., 3., 4.]]
Projected q: [[-2.34, -1.44,  2.75,  1.23, -0.56,  0.78]]
```

Here, the 4-D token is projected into a 6-D vector using `W_q`.  
This new representation can now be partitioned across heads in multi-head attention.
