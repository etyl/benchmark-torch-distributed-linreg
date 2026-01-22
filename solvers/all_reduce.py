from benchopt.stopping_criterion import SufficientProgressCriterion
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from benchmark_utils.mpi_solver import DistributedSolver
from benchmark_utils.dataset_utils import get_dataloader


class Solver(DistributedSolver):
    name = "all-reduce"

    parameters = {
        "n_workers": [1, 16],
        "batch_size": [32],
        "lr": [1e-3],
        "moments": [False]
    }

    requirements = ["numpy", "torch"]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-10, patience=3, strategy="iteration"
    )

    @classmethod
    def init_worker(cls, args, rank, world_size):
        """
        Initialize the worker environment, clear logs, and load data.
        Returns the local data tensor (X_local).
        """
        dataloader = get_dataloader(
            args.x_path, args.y_path, args.batch_size
        )
        model = nn.Linear(
            dataloader.dataset.X.shape[1],
            dataloader.dataset.Y.shape[1],
            bias=False
        )
        if args.device == "cpu":
            model = DDP(model)
        elif args.device.startswith("cuda"):
            local_rank = int(os.environ["LOCAL_RANK"])
            model = model.to(local_rank)
            model = DDP(model, device_ids=[local_rank])
        else:
            raise ValueError(f"Unsupported device: {args.device}")

        return dataloader, model

    @classmethod
    def worker_run(
        cls, n_iter, worker_ctx, args, rank, world_size
    ):
        dataloader, model = worker_ctx

        if args.moments:
            optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optim = torch.optim.SGD(model.parameters(), lr=args.lr)

        criterion = nn.MSELoss()

        for k, (x, y) in enumerate(dataloader):
            optim.zero_grad()

            if k >= n_iter:
                break

            # Local Computation
            y_pred = model(x.to(model.device))
            loss = criterion(y_pred, y.to(model.device))

            loss.backward()
            optim.step()

        return dict(model=model.module.to("cpu"))


if __name__ == "__main__":
    Solver.entry_point()
