from torch.utils.data import DataLoader, Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, data, seq_len):
        if isinstance(data, list):
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)
        elif isinstance(data, torch.Tensor):
            data = data.float().unsqueeze(-1) if data.ndim == 1 else data.float()
        else:  # assume numpy
            data = torch.from_numpy(data).float().unsqueeze(-1)

        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1  # corrected to include last window

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len]

# Use it
series = [1,2,3,4,5,6,7,8]
dataset = SlidingWindowDataset(series, seq_len=3)

for i in range(len(dataset)):
    print(dataset[i])

'''
Outout: 
tensor([[1.],
        [2.],
        [3.]])
tensor([[2.],
        [3.],
        [4.]])
tensor([[3.],
        [4.],
        [5.]])
tensor([[4.],
        [5.],
        [6.]])
tensor([[5.],
        [6.],
        [7.]])
tensor([[6.],
        [7.],
        [8.]])

'''

'''
Without the class: you explicitly write the loop for i in range(...) and slice series[i:i+seq_len].

With the class: you just define how to slice once in __getitem__, and PyTorch automatically handles indexing and batching for you.
'''