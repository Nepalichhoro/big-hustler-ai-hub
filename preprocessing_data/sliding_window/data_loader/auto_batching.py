from torch.utils.data import Dataset, DataLoader

class SlidingWindowDataset(Dataset):
    def __init__(self, data, seq_len):
        if isinstance(data, list):
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)
        elif isinstance(data, torch.Tensor):
            data = data.float().unsqueeze(-1) if data.ndim == 1 else data.float()
        else:  # numpy
            data = torch.from_numpy(data).float().unsqueeze(-1)
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1  # include last window

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len]

# Create dataset and loader
series = [1,2,3,4,5,6,7,8]
dataset = SlidingWindowDataset(series, seq_len=3)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

# Iterate
for batch in loader:
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