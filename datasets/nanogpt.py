from benchopt import BaseDataset
from benchopt.config import get_data_path

import glob
from bisect import bisect_right
from pathlib import Path

import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

from benchmark_utils.model_gpt2 import GPT, GPTConfig


def download_data(data_dir, n_chunks=104):

    # Download the GPT-2 tokens of Fineweb10B from huggingface. This
    # saves about an hour of startup time compared to regenerating them.
    for i in range(n_chunks):
        chunk = "val" if i == 0 else "train"
        fname = f"fineweb_{chunk}_{i:06d}.bin"
        if not (data_dir / fname).exists():
            hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                            repo_type="dataset", local_dir=data_dir)


def _read_shard_header(file):
    """Return the header tensor for a shard after a basic integrity check."""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    return header


def _load_data_shard(file):
    header = _read_shard_header(file)
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        pin_memory = torch.cuda.is_available()
        # avoid pin_memory copy by @YouJiacheng
        tokens = torch.empty(
            num_tokens, dtype=torch.uint16, pin_memory=pin_memory
        )
        f.seek(256 * 4)
        # avoid bytes->array copy by @YouJiacheng
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, (
            "number of tokens read does not match header"
        )
    return tokens


class FinewebDataset(Dataset):
    """Map-style dataset built from FineWeb GPT-2 token shards."""

    def __init__(self, filename_pattern, seq_len=1024, max_tokens=None):
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
        if not self.files:
            raise FileNotFoundError(
                f"No shards found for pattern {filename_pattern!r}"
            )

        self.file_metas = []
        total_tokens = 0
        total_sequences = 0
        for file in self.files:
            header = _read_shard_header(file)
            shard_tokens = int(header[2])
            if self.max_tokens is not None:
                remaining = self.max_tokens - total_tokens
                if remaining <= 0:
                    break
                shard_tokens = min(shard_tokens, remaining)

            sequence_count = max((shard_tokens - 1) // self.seq_len, 0)
            if sequence_count == 0:
                continue

            self.file_metas.append({
                "path": file,
                "num_tokens": shard_tokens,
                "num_sequences": sequence_count,
            })
            total_sequences += sequence_count
            total_tokens += shard_tokens

            if self.max_tokens is not None and total_tokens >= self.max_tokens:
                break

        if total_sequences == 0:
            raise ValueError("No usable sequences found in provided shards.")

        self._length = total_sequences
        self._prefix = []
        running = 0
        for meta in self.file_metas:
            running += meta["num_sequences"]
            self._prefix.append(running)

        self._cache = {"file_idx": None, "tokens": None}

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index < 0 or index >= self._length:
            raise IndexError("Index out of range for FinewebDataset")

        file_idx = bisect_right(self._prefix, index)
        prev = self._prefix[file_idx - 1] if file_idx > 0 else 0
        seq_idx = index - prev

        tokens = self._get_tokens(file_idx)
        start = seq_idx * self.seq_len
        end = start + self.seq_len + 1
        buf = tokens[start:end]
        inputs = buf[:-1].to(dtype=torch.int32)
        targets = buf[1:].to(dtype=torch.int64)
        return inputs, targets

    def _get_tokens(self, file_idx):
        if self._cache["file_idx"] != file_idx:
            tokens = _load_data_shard(self.file_metas[file_idx]["path"])
            num_tokens = self.file_metas[file_idx]["num_tokens"]
            self._cache = {
                "file_idx": file_idx,
                "tokens": tokens[:num_tokens],
            }
        return self._cache["tokens"]


class Dataset(BaseDataset):

    name = "nanogpt"
    parameters = {
        'n_chunks': [4],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = ["huggingface_hub"]

    def get_data(self):
        print("Getting data")
        data_dir = get_data_path("fineweb10B")
        download_data(data_dir, n_chunks=self.n_chunks)

        # from scratch (random weights)
        config = GPTConfig(
            vocab_size=50304, n_layer=12, n_head=12, n_embd=768,
            # max_seq_len=4*64*1024 - This is for Rotary Positional Embedding
        )
        model = GPT(config)
        print("Returning data")

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            dataset=FinewebDataset(
                str(data_dir / "fineweb_train_*.bin")
            ),
            model=model,
        )