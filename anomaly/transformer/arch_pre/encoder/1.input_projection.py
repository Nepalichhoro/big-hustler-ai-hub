import torch
import torch.nn as nn

proj = nn.Linear(4, 8)   # from 4 features -> d_model=8
x = torch.tensor([[[20000.0, 20010.0, 19990.0, 100.0]]]) # (1,1,4)

out = proj(x)
print("Input:", x.shape, x) # ([1, 1, 4])
print("Projected:", out.shape, out) # (1,1,8)


'''

