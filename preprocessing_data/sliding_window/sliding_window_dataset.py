import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SlidingWindowDataset(Dataset):
    def __init__(self, data, seq_len):
        """
        Args:
            data (array-like): 1D numpy array or list of time series values
            seq_len (int): Length of each window
        """
        if isinstance(data, list):
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)
        elif isinstance(data, torch.Tensor):
            data = data.float().unsqueeze(-1) if data.ndim == 1 else data.float()
        else:  # assume numpy
            data = torch.from_numpy(data).float().unsqueeze(-1)

        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len]  # shape: (seq_len, 1)

# Example 

series = [1,2,3,4,5,6,7,8]
dataset = SlidingWindowDataset(series, seq_len=3)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch in loader:
    print(batch.shape)  # (batch_size, seq_len, 1)