from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class HOGATLayer(nn.Module):
    """High-order neighbor propagation attention layer.

    Implements Eq.(13–15)-style multi-hop aggregation by separately attending to
    neighbors within each hop distance up to k_hops, then combining.
    """
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, k_hops: int = 3, dropout: float = 0.1):
        super().__init__()
        self.k_hops = k_hops
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.ModuleList([nn.Linear(in_dim, out_dim * heads, bias=False) for _ in range(k_hops)])
        self.attn = nn.ParameterList([nn.Parameter(torch.randn(heads, 2*out_dim)) for _ in range(k_hops)])
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        # adjs[l] is adjacency for l+1 hops, each [N,N] binary
        outs = []
        N = x.size(0)
        for l in range(self.k_hops):
            adj = adjs[l]
            h = self.proj[l](x).view(N, self.heads, -1)
            h_i = h.unsqueeze(1).repeat(1, N, 1, 1)
            h_j = h.unsqueeze(0).repeat(N, 1, 1, 1)
            cat = torch.cat([h_i, h_j], dim=-1)
            e = self.leaky((cat * self.attn[l].view(1,1,self.heads,-1)).sum(dim=-1))
            e = e.masked_fill(adj.unsqueeze(-1) == 0, float("-inf"))
            a = torch.softmax(e, dim=1)
            a = self.dropout(a)
            out = (a.unsqueeze(-1) * h_j).sum(dim=1)  # [N,H,F']
            outs.append(out.reshape(N, -1))
        return torch.cat(outs, dim=-1)

class HOGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, heads: int = 4, k_hops: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layer = HOGATLayer(in_dim, hidden_dim // heads, heads=heads, k_hops=k_hops, dropout=dropout)
        self.out_dim = (hidden_dim) * k_hops

    def forward(self, x: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        return F.relu(self.layer(x, adjs))
