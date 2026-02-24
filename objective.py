from benchopt import BaseObjective
import numpy as np
import torch


class Objective(BaseObjective):
    name = "Linear Regression"
    min_benchopt_version = "1.7"

    parameters = {
        "device": ["cpu"],
        "slurm_nodes": [1],
        "slurm_gpus_per_node": [2]
    }

    def set_data(self, X, Y):
        self.X = X
        self.Y = Y

    def get_one_result(self):
        return dict(model=torch.nn.Linear(self.X.shape[1], self.Y.shape[1], bias=False))

    def evaluate_result(self, model, logs={}):
        return {
            k: sum(v) for k, v in logs.items()
        }

    def get_objective(self):
        return dict(
            X=self.X, Y=self.Y, device=self.device
        )
