from benchopt import BaseObjective
import numpy as np
from numpy.lib.format import open_memmap
import torch


@torch.no_grad()
def _compute_loss(X, y, model):
    residuals = model(X.to(torch.float32)) - y.to(torch.float32)
    loss = torch.mean(residuals ** 2).item()
    return loss


class Objective(BaseObjective):
    name = "Linear Regression"
    min_benchopt_version = "1.7"

    parameters = {
        "device": ["cpu"]
    }

    def set_data(self, x_path, y_path):
        self.x_path, self.y_path = x_path, y_path

    def get_one_result(self):
        x = open_memmap(self.x_path)
        y = open_memmap(self.y_path)
        return dict(model=torch.nn.Linear(x.shape[1], y.shape[1], bias=False))

    def evaluate_result(self, model):
        x = torch.from_numpy(np.load(self.x_path))
        y = torch.from_numpy(np.load(self.y_path))
        train_loss = _compute_loss(
            x, y, model
        )
        return {
            "value": train_loss,
        }

    def get_objective(self):
        return dict(
            x_path=self.x_path, y_path=self.y_path, device=self.device
        )
