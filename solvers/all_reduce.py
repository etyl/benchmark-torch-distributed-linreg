from collections import defaultdict
import time
from benchopt import BaseSolver
import os
import torch
import torch.nn as nn
import torch.distributed as dist

from benchmark_utils.dataset_utils import get_dataloader


def setup_distributed(device):
    """Maps SLURM variables to PyTorch DDP variables and initializes the process group."""
    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    if device.startswith("cuda"):
        dist.init_process_group(backend="nccl", init_method="env://")
    elif device == "cpu":
        dist.init_process_group(backend="gloo", init_method="env://")
    else:
        raise ValueError(f"Unsupported device: {device}")



class Solver(BaseSolver):
    name = "all-reduce"

    parameters = {
        "batch_size": [32],
        "lr": [1e-3],
    }

    requirements = ["numpy", "pytorch:pytorch"]

    sampling_strategy = "run_once"

    def set_objective(self, x_path, y_path, device):
        self.device = device
        setup_distributed(device)
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        self.dataloader = get_dataloader(
            x_path, y_path, int(self.batch_size)
        )
        self.model = nn.Linear(
            self.dataloader.dataset.X.shape[1],
            self.dataloader.dataset.Y.shape[1],
            bias=False,
        ).to(local_rank)

    def run(self, _):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

        optim = torch.optim.Adam(self.model.parameters(), lr=float(self.lr))
        criterion = nn.MSELoss()

        world_size = int(os.environ["WORLD_SIZE"])

        self.logs = defaultdict(list)
        k = 0
        while True:
            for x, y in self.dataloader:
                optim.zero_grad()

                k += 1
                if k > 100:
                    dist.destroy_process_group()
                    return

                y_pred = self.model(x.to(self.device))
                loss = criterion(y_pred, y.to(self.device))

                loss.backward()

                # Synchronize gradients
                if use_cuda:
                    start.record()
                else:
                    t0 = time.perf_counter()
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= world_size
                if use_cuda:
                    end.record()
                    torch.cuda.synchronize()
                    self.logs["comm_time"].append(start.elapsed_time(end)/1000)
                else:
                    self.logs["comm_time"].append(time.perf_counter() - t0)

                optim.step()

    def get_result(self):
        return dict(
            model=self.model.cpu(),
            logs=self.logs
        )
