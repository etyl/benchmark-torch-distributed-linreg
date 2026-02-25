from benchopt import BaseObjective
import numpy as np
import torch


class Objective(BaseObjective):
    name = "Linear Regression"
    min_benchopt_version = "1.8"

    parameters = {
        "device": ["cpu"],
    }

    def set_data(self, X, Y):
        self.X = X
        self.Y = Y

    def get_one_result(self):
        return dict(model=torch.nn.Linear(self.X.shape[1], self.Y.shape[1], bias=False))

    def evaluate_result(self, model, logs={}):
        mean_logs = {
            k: sum(v) for k, v in logs.items()
        }
        return {
            "comm_ratio": mean_logs["comm_time"] / mean_logs["run_time"],
            **mean_logs
        }

    def get_objective(self):
        return dict(
            X=self.X, Y=self.Y, device=self.device
        )
