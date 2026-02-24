from benchopt import BaseDataset
from benchmark_utils import get_data_path
import numpy as np
import os


class Dataset(BaseDataset):
    name = "simulated"

    parameters = {
        'n': [16*1024],
        'd1': [400],
        'd2': [400],
    }
    requirements = ["numpy"]

    def get_data(self):
        rng = np.random.RandomState(42)

        X = rng.randn(self.n, self.d1)
        W_linear = rng.randn(self.d1, self.d2)
        Y = X @ W_linear

        return dict(X=X, Y=Y)
