import torch

series = [1,2,3,4,5,6,7,8]
seq_len = 3

# Make sliding windows manually
windows = []
for i in range(len(series) - seq_len + 1):
    window = torch.tensor(series[i:i+seq_len], dtype=torch.float32).unsqueeze(-1)
    windows.append(window)

batch_size = 2
for i in range(0, len(windows), batch_size):
    batch = torch.stack(windows[i:i+batch_size])  # stack windows into one tensor
    print("Batch shape:", batch.shape)
    print(batch)

'''
Batch shape: torch.Size([2, 3, 1])
tensor([[[1.],
         [2.],
         [3.]],

        [[2.],
         [3.],
         [4.]]])

Batch shape: torch.Size([2, 3, 1])
tensor([[[3.],
         [4.],
         [5.]],

        [[4.],
         [5.],
         [6.]]])

Batch shape: torch.Size([2, 3, 1])
tensor([[[5.],
         [6.],
         [7.]],

        [[6.],
         [7.],
         [8.]]])

'''

'''

Manual batching = you loop, slice, and stack yourself.

DataLoader batching = you just say batch_size=2, and it does all that automatically.

It also supports shuffling (shuffle=True) and parallel loading (num_workers > 0).


So now you can train a model by just writing:

for batch in loader:
    # batch is already ready: shape (batch_size, seq_len, features)
'''