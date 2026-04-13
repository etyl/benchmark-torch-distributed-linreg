import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


def _is_memory_error(err):
    msg = str(err).lower()
    return (
        isinstance(err, MemoryError)
        or "out of memory" in msg
        or "cuda out of memory" in msg
        or "cudnn_status_not_supported" in msg
    )


def _clear_probe_state(model, device):
    model.zero_grad(set_to_none=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def _probe_batch_size(model, dataset, batch_size, device):
    ok = True
    try:
        probe_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        batch = next(iter(probe_loader))
        batch = [x.to(device) for x in batch]
        model.zero_grad(set_to_none=True)
        loss, *_ = model(*batch)
        loss.backward()
    except (RuntimeError, MemoryError) as err:
        if not _is_memory_error(err):
            raise
        ok = False
    finally:
        _clear_probe_state(model, device)
    return ok

def get_max_batch_size(model, dataset, device):
    batch_size = 4
    last_valid = 0

    while True:
        print(f"Probing batch_size={batch_size+1} on rank {dist.get_rank()}")
        batch_valid = _probe_batch_size(model, dataset, batch_size+1, device)

        if not batch_valid:
            break

        last_valid = batch_size
        batch_size *= 2

    if last_valid == 0:
        raise RuntimeError(
            "Unable to run even the initial local_batch_size. "
            "Try lowering local_batch_size in solver parameters."
        )

    if dist.get_rank() == 0:
        print(f"Using maximized local_batch_size={last_valid}")

    return last_valid
