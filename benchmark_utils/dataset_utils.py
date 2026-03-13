from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_dataloader(dataset, batch_size):
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0
    )
    return dataloader
