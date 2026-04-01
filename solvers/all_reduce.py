from collections import defaultdict
import time
from benchopt import BaseSolver
import os
import torch
import torch.distributed as dist

from benchmark_utils.dataset_utils import get_dataloader


def setup_distributed(device):
    """Maps SLURM variables to PyTorch DDP variables and initializes the process group."""
    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # NCCL debugging environment variables
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,TUNING"

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
        "local_batch_size": [32],
        "lr": [1e-3],
        "slurm_nodes": [2]
    }

    requirements = ["pytorch:pytorch"]

    sampling_strategy = "run_once"

    def set_objective(self, dataset, model, device):
        self.device = device
        self.dataset = dataset
        self.model = model

    def run(self, _):
        setup_distributed(self.device)
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        model = self.model.to(device=self.device)
        dataloader = get_dataloader(self.dataset, batch_size=self.local_batch_size)

        use_cuda = self.device.startswith("cuda")
        if use_cuda:
            start_run = torch.cuda.Event(enable_timing=True)
            start_com = torch.cuda.Event(enable_timing=True)
            end_run = torch.cuda.Event(enable_timing=True)
            end_com = torch.cuda.Event(enable_timing=True)

        optim = torch.optim.Adam(model.parameters(), lr=float(self.lr))

        self.logs = defaultdict(list)

        for batch in dataloader:
            optim.zero_grad()

            batch = [x.to(self.device) for x in batch]
            loss, *_ = model(*batch)
            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= world_size

            optim.step()
            break

        if use_cuda:
            torch.cuda.synchronize()
            dist.barrier()
            start_run.record()
        else:
            t0_run = time.perf_counter()

        k = 0
        stop_training = False
        while not stop_training:
            for batch in dataloader:
                print(f"Rank {dist.get_rank()} - Batch {k}")

                optim.zero_grad()

                batch = [x.to(self.device) for x in batch]
                loss, *_ = model(*batch)
                loss.backward()

                # Synchronize gradients
                if use_cuda:
                    torch.cuda.synchronize()
                    dist.barrier()
                    start_com.record()
                t0_com = time.perf_counter()

                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= world_size

                if use_cuda:
                    end_com.record()
                    torch.cuda.synchronize()
                    self.logs["comm_time"].append(start_com.elapsed_time(end_com)/1000)
                else:
                    self.logs["comm_time"].append(time.perf_counter() - t0_com)
                self.logs["comm_time_cpu"].append(time.perf_counter() - t0_com)

                optim.step()

                k += 1
                if k > 40:
                    stop_training = True
                    break

        if use_cuda:
            end_run.record()
            torch.cuda.synchronize()
            self.logs["run_time"].append(start_run.elapsed_time(end_run)/1000)
        else:
            self.logs["run_time"].append(time.perf_counter() - t0_run)

    def get_result(self):
        return dict(
            logs=self.logs
        )
