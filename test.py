import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# 1. Dummy Dataset for testing
class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.x = torch.randn(size, 10)
        self.y = torch.randn(size, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def setup_distributed():
    """Maps SLURM variables to PyTorch DDP variables and initializes the process group."""
    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # Initialize the process group (NCCL is the standard backend for GPUs)
    dist.init_process_group(backend="nccl", init_method="env://")

    # Set the specific GPU for this process
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    # Initialize DDP
    local_rank = setup_distributed()
    global_rank = int(os.environ["RANK"])

    # Print only on the master process to avoid terminal clutter
    if global_rank == 0:
        print(f"Starting training on {os.environ['WORLD_SIZE']} GPUs!")

    # 2. Create Model and move it to the correct GPU
    model = nn.Linear(10, 1).to(local_rank)

    # 3. Setup DataLoader with DistributedSampler
    dataset = DummyDataset()
    sampler = DistributedSampler(dataset, shuffle=True)

    # Note: shuffle must be False in DataLoader when using DistributedSampler
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 4. Training Loop
    epochs = 5
    for epoch in range(epochs):
        # Crucial: set the epoch on the sampler to ensure proper shuffling across epochs
        sampler.set_epoch(epoch)

        for batch_idx, (data, targets) in enumerate(dataloader):
            print(f"Global Rank: {global_rank}, Local Rank: {local_rank}, Batch: {batch_idx}")
            data = data.to(local_rank)
            targets = targets.to(local_rank)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= int(os.environ["WORLD_SIZE"])

            optimizer.step()

        if global_rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}] completed. Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    main()