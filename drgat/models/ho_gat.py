
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, JumpingKnowledge, global_mean_pool

class HOGAT(nn.Module):
    def __init__(self, in_dim, hidden=128, heads=4, layers=3, jk_mode="cat", num_classes=2):
        super().__init__()
        self.gats = nn.ModuleList()
        self.gats.append(GATConv(in_dim, hidden, heads=heads, concat=True, dropout=0.2))
        for _ in range(layers-1):
            self.gats.append(GATConv(hidden*heads, hidden, heads=heads, concat=True, dropout=0.2))
        self.jk = JumpingKnowledge(jk_mode, channels=hidden*heads, num_layers=layers)
        out_dim = hidden*heads*layers if jk_mode == "cat" else hidden*heads
        self.pred = nn.Sequential(
            nn.Linear(out_dim+1, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, edge_index, batch, inv_dist_weight=None):
        feats = []
        h = x
        for i, conv in enumerate(self.gats):
            h = conv(h, edge_index)
            h = torch.relu(h)
            feats.append(h)
        h = self.jk(feats)
        g = global_mean_pool(h, batch)
        if inv_dist_weight is None:
            inv_dist_weight = torch.ones((g.size(0), 1), device=g.device)
        out = torch.cat([g, inv_dist_weight], dim=-1)
        return self.pred(out)
