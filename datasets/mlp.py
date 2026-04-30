from benchopt import BaseDataset
import numpy as np
import torch.nn as nn
import torch


class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, d):
        self.d = d
        rng = np.random.RandomState(42)
        self.W_linear = rng.randn(self.d, self.d)

    def __len__(self):
        return 100_000_000

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        X = torch.randn(len(idx), self.d)
        Y = torch.randn_like(X)
        return X, Y


class MLP(nn.Module):
    def __init__(self, d, layers=1, bias=False):
        super().__init__()
        self.d = d
        self.bias = bias
        layer_list = []
        for _ in range(layers):
            layer_list.append(nn.Linear(self.d, self.d, bias=self.bias))
            layer_list.append(nn.ReLU())
        self.model = nn.Sequential(*layer_list)
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        return loss, None


class Dataset(BaseDataset):
    name = "mlp"

    parameters = {
        'd': [400],
        'layers': [1],
    }
    requirements = ["numpy"]

    def get_data(self):
        dataset = MLPDataset(self.d)
        model = MLP(self.d, self.layers)
        return dict(dataset=dataset, model=model)
