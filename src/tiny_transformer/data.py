from __future__ import annotations

import torch


class TextDataset:
    def __init__(self, token_ids: list[int], block_size: int, device: str) -> None:
        if len(token_ids) <= block_size:
            raise ValueError("Dataset must contain more tokens than block_size")
        self.data = torch.tensor(token_ids, dtype=torch.long, device=device)
        self.block_size = block_size
        self.device = device

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.block_size
        starts = torch.randint(0, max_start, (batch_size,), device=self.device)
        x = torch.stack([self.data[start : start + self.block_size] for start in starts])
        y = torch.stack([self.data[start + 1 : start + self.block_size + 1] for start in starts])
        return x, y


def split_tokens(token_ids: list[int], train_fraction: float = 0.9) -> tuple[list[int], list[int]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split_idx = int(len(token_ids) * train_fraction)
    return token_ids[:split_idx], token_ids[split_idx:]
