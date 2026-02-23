from collections import defaultdict
import time

from benchopt.stopping_criterion import SufficientProgressCriterion
import os
import torch
import torch.nn as nn
import torch.distributed as dist

from benchmark_utils.mpi_solver import DistributedSolver
from benchmark_utils.dataset_utils import get_dataloader


class Solver(DistributedSolver):
    name = "all-reduce"

    parameters = {
        "n_workers": [1, 16],
        "batch_size": [32],
        "lr": [1e-3],
        "adam": ["true", "false"]
    }

    requirements = ["numpy", "torch"]

    sampling_strategy = "run_once"

    @classmethod
    def init_worker(cls, args, rank, world_size):
        torch.manual_seed(0)
        dataloader = get_dataloader(
            args.x_path, args.y_path, args.batch_size
        )
        model = nn.Linear(
            dataloader.dataset.X.shape[1],
            dataloader.dataset.Y.shape[1],
            bias=False,
        )

        if args.device == "cpu":
            pass
        elif args.device.startswith("cuda"):
            local_rank = int(os.environ["LOCAL_RANK"])
            model = model.to(local_rank)
        else:
            raise ValueError(f"Unsupported device: {args.device}")

        return dataloader, model

    @classmethod
    def worker_run(
        cls, n_iter, worker_ctx, args, rank, world_size
    ):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        dataloader, model = worker_ctx

        if args.adam == "true":
            optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optim = torch.optim.SGD(model.parameters(), lr=args.lr)

        criterion = nn.MSELoss()

        device = next(model.parameters()).device

        logs = defaultdict(list)
        k = 0
        while True:
            for x, y in dataloader:
                optim.zero_grad()

                k += 1
                if k > 100:
                    return dict(model=model.to("cpu"), logs=logs)

                y_pred = model(x.to(device))
                loss = criterion(y_pred, y.to(device))

                loss.backward()

                # Synchronize gradients
                if use_cuda:
                    start.record()
                else:
                    t0 = time.perf_counter()
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= world_size
                if use_cuda:
                    end.record()
                    torch.cuda.synchronize()
                    logs["comm_time"].append(start.elapsed_time(end)/1000)
                else:
                    logs["comm_time"].append(time.perf_counter() - t0)

                optim.step()


if __name__ == "__main__":
    Solver.entry_point()
