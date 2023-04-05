from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGraphAttention(nn.Module):
    """Minimal attention layer for dense adjacency matrices.

    NOTE: For performance, replace with torch_geometric.nn.GATConv.
    """
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn = nn.Parameter(torch.randn(heads, 2*out_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [N, F], adj: [N, N] (0/1)
        N = x.size(0)
        h = self.W(x).view(N, self.heads, -1)  # [N, H, F']
        # compute attention scores e_ij per head
        # naive O(N^2); ok for small pathway graphs
        h_i = h.unsqueeze(1).repeat(1, N, 1, 1)  # [N, N, H, F']
        h_j = h.unsqueeze(0).repeat(N, 1, 1, 1)  # [N, N, H, F']
        cat = torch.cat([h_i, h_j], dim=-1)       # [N, N, H, 2F']
        e = self.leaky((cat * self.attn.view(1,1,self.heads,-1)).sum(dim=-1))  # [N,N,H]
        e = e.masked_fill(adj.unsqueeze(-1) == 0, float("-inf"))
        a = torch.softmax(e, dim=1)  # normalize over neighbors j
        a = self.dropout(a)
        out = (a.unsqueeze(-1) * h_j).sum(dim=1)  # [N,H,F']
        return out.reshape(N, -1)

class GraphAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.enc_attn = SimpleGraphAttention(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.enc_proj = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.enc_attn(x, adj)
        z = self.enc_proj(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, adj)
        xhat = self.decode(z)
        return z, xhat
