from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hogat import HOGAT

class DRGATPredictor(nn.Module):
    """Applies HO-GAT on each pathway subgraph, readout, then concatenates distance features.

    Mirrors Eq.(16–17) conceptually.
    """
    def __init__(self, node_in: int, hogat_hidden: int, heads: int, k_hops: int, mlp_hidden: list[int], dropout: float = 0.2):
        super().__init__()
        self.hogat = HOGAT(node_in, hogat_hidden, heads=heads, k_hops=k_hops, dropout=dropout)
        in_dim = self.hogat.out_dim + 1  # + distance scalar per pathway
        mlp = []
        d = in_dim
        for h in mlp_hidden:
            mlp += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        mlp += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*mlp)

    def readout(self, H: torch.Tensor) -> torch.Tensor:
        # simple mean pooling
        return H.mean(dim=0)

    def forward(self, pathways_batch: list[dict]) -> torch.Tensor:
        """pathways_batch: list of dicts, each has keys:
            - x: [N_i, F]
            - adjs: list of [N_i,N_i] for hops
            - dist: float
        returns logits [B]
        """
        feats = []
        for p in pathways_batch:
            H = self.hogat(p["x"], p["adjs"])
            r = self.readout(H)
            dist = torch.tensor([p["dist"]], device=r.device, dtype=r.dtype)
            feats.append(torch.cat([r, dist], dim=0))
        Z = torch.stack(feats, dim=0)
        logits = self.mlp(Z).squeeze(-1)
        return logits
