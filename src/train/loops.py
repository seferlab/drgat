from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch
from torch.utils.data import DataLoader

@dataclass
class EarlyStopping:
    patience: int = 10
    best: float = float("inf")
    bad: int = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - 1e-6:
            self.best = val_loss
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience

def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        optim.zero_grad(set_to_none=True)
        batch = move_batch(batch, device)
        loss = loss_fn(model, batch)
        loss.backward()
        optim.step()
        total += float(loss.item()) * batch_size(batch)
        n += batch_size(batch)
    return total / max(n, 1)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        loss = loss_fn(model, batch)
        total += float(loss.item()) * batch_size(batch)
        n += batch_size(batch)
    return total / max(n, 1)

def move_batch(batch, device):
    # handle common patterns
    if isinstance(batch, dict):
        out = {}
        for k,v in batch.items():
            if torch.is_tensor(v): out[k] = v.to(device)
            else: out[k] = v
        return out
    return batch

def batch_size(batch):
    if isinstance(batch, dict) and "y" in batch and hasattr(batch["y"], "shape"):
        return int(batch["y"].shape[0])
    if hasattr(batch, "__len__"):
        return len(batch)
    return 1
