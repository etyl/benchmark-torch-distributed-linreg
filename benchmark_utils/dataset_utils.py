import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class TorchDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.Y[idx], dtype=torch.float32)
        )


def get_dataloader(x_path, y_path, batch_size):
    X = np.load(x_path, mmap_mode='r')
    Y = np.load(y_path, mmap_mode='r')
    dataset = TorchDataset(X, Y)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0
    )
    return dataloader
