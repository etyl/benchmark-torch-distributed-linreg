from benchopt import BaseDataset
import numpy as np
import torch.nn as nn
import torch


class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, d, n):
        self.d = d
        self.n = n
        rng = np.random.RandomState(42)
        self.W_linear = rng.randn(self.d, self.d)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        X = torch.randn(len(idx), self.d)
        Y = torch.randn_like(X)
        return X, Y


class Dataset(BaseDataset):
    name = "mlp"

    parameters = {
        'n': [16*1024],
        'd': [400],
        'layers': [1],
        'bias': [False],
    }
    requirements = ["numpy"]

    def get_data(self):
        dataset = MLPDataset(self.d, self.n)

        layers = []
        for _ in range(self.layers):
            layers.append(nn.Linear(self.d, self.d, bias=self.bias))
            layers.append(nn.ReLU())
        model = nn.Sequential(*layers)

        return dict(dataset=dataset, model=model)
