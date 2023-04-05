import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class SimpleGAT(nn.Module):
    """A lightweight GAT baseline (used for 'No HO-GAT' ablation).

    Differences vs HOGAT:
      - no JumpingKnowledge concatenation
      - uses the last layer output only
    """
    def __init__(self, in_dim: int, hidden: int = 128, heads: int = 4, layers: int = 3, num_classes: int = 2):
        super().__init__()
        self.gats = nn.ModuleList()
        self.gats.append(GATConv(in_dim, hidden, heads=heads, concat=True, dropout=0.2))
        for _ in range(layers - 1):
            self.gats.append(GATConv(hidden * heads, hidden, heads=heads, concat=True, dropout=0.2))
        out_dim = hidden * heads
        self.pred = nn.Sequential(
            nn.Linear(out_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, edge_index, batch, inv_dist_weight=None):
        h = x
        for conv in self.gats:
            h = torch.relu(conv(h, edge_index))
        g = global_mean_pool(h, batch)
        if inv_dist_weight is None:
            inv_dist_weight = torch.ones((g.size(0), 1), device=g.device)
        out = torch.cat([g, inv_dist_weight], dim=-1)
        return self.pred(out)
