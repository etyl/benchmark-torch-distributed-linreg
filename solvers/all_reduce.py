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

    # Check if dist is already initialized to avoid reinitialization
    if dist.is_initialized():
        return

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
        "slurm_nodes": [1],
        "slurm_gpus_per_node": [2]
    }

    requirements = ["pytorch:pytorch"]

    sampling_strategy = "run_once"

    def set_objective(self, X, Y, device):
        self.X = X
        self.Y = Y
        self.device = device
        setup_distributed(device)
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        self.dataloader = get_dataloader(
            X, Y, int(self.batch_size)
        )
        self.model = nn.Linear(
            self.dataloader.dataset.X.shape[1],
            self.dataloader.dataset.Y.shape[1],
            bias=False,
        ).to(local_rank)

    def run(self, _):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            start_run = torch.cuda.Event(enable_timing=True)
            start_com = torch.cuda.Event(enable_timing=True)
            end_run = torch.cuda.Event(enable_timing=True)
            end_com = torch.cuda.Event(enable_timing=True)

        optim = torch.optim.Adam(self.model.parameters(), lr=float(self.lr))
        criterion = nn.MSELoss()

        world_size = int(os.environ["WORLD_SIZE"])

        self.logs = defaultdict(list)

        if use_cuda:
            torch.cuda.synchronize()
            start_run.record()
        else:
            t0_run = time.perf_counter()

        k = 0
        while True:
            for x, y in self.dataloader:

                optim.zero_grad()

                k += 1
                if k > 100:
                    if use_cuda:
                        end_run.record()
                        torch.cuda.synchronize()
                        self.logs["run_time"].append(start_run.elapsed_time(end_run)/1000)
                    else:
                        self.logs["run_time"].append(time.perf_counter() - t0_run)
                        dist.destroy_process_group()
                    return

                y_pred = self.model(x.to(self.device))
                loss = criterion(y_pred, y.to(self.device))

                loss.backward()

                # Synchronize gradients
                if use_cuda:
                    torch.cuda.synchronize()
                    start_com.record()
                else:
                    t0_com = time.perf_counter()

                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= world_size

                if use_cuda:
                    end_com.record()
                    torch.cuda.synchronize()
                    self.logs["comm_time"].append(start_com.elapsed_time(end_com)/1000)
                else:
                    self.logs["comm_time"].append(time.perf_counter() - t0_com)

                optim.step()

    def get_result(self):
        return dict(
            model=self.model.cpu(),
            logs=self.logs
        )
