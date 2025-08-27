import torch

series = [1,2,3,4,5,6,7,8]
seq_len = 3

# Manually create sliding windows
windows = []
for i in range(len(series) - seq_len + 1):
    window = torch.tensor(series[i:i+seq_len], dtype=torch.float32).unsqueeze(-1)
    windows.append(window)

for w in windows:
    print(w)

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