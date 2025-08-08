
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GraphAutoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, latent_dim=64, heads=2):
        super().__init__()
        self.enc1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=0.1)
        self.enc2 = GATConv(hidden_dim*heads, latent_dim, heads=1, concat=True, dropout=0.1)
        self.post = nn.Linear(latent_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def encode(self, x, edge_index):
        h = self.enc1(x, edge_index)
        h = torch.relu(h)
        h = self.enc2(h, edge_index)
        z = self.post(torch.relu(h))
        return z

    def decode(self, z):
        return self.dec(z)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        xhat = self.decode(z)
        return xhat, z
