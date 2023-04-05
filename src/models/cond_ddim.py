from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_beta_schedule(T: int, schedule: str = "linear", beta_start: float = 1e-4, beta_end: float = 2e-2):
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)
    raise ValueError(f"Unknown schedule: {schedule}")

class CondEpsModel(nn.Module):
    """Backbone epsilon_theta(x_t, t, c) using affine layers (not U-Net).

    Matches the paper's statement that convolution layers are replaced by affine layers for tabular latent data.
    """
    def __init__(self, dim: int, cond_dim: int = 16, time_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_embed = nn.Embedding(2, cond_dim)  # binary condition: resistant/sensitive
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim + cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # t: [B] int, normalize to [0,1]
        t_norm = (t.float() / (t.max().clamp(min=1))).unsqueeze(-1)
        te = self.time_embed(t_norm)
        ce = self.cond_embed(c.long())
        inp = torch.cat([x_t, te, ce], dim=-1)
        return self.net(inp)

class ConditionalDDIM(nn.Module):
    def __init__(self, dim: int, T: int = 1000, schedule: str = "linear", cond_dim: int = 16):
        super().__init__()
        self.T = T
        betas = make_beta_schedule(T, schedule)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.eps_model = CondEpsModel(dim=dim, cond_dim=cond_dim)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a = self.alpha_bar[t].unsqueeze(-1)
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * eps

    def loss(self, x0: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device)
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps)
        eps_hat = self.eps_model(x_t, t, c)
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def sample(self, n: int, c: int, steps: int = 50, eta: float = 0.0, device: str | torch.device = "cpu") -> torch.Tensor:
        """DDIM sampling (deterministic if eta=0)"""
        device = torch.device(device)
        x = torch.randn(n, self.eps_model.net[-1].out_features, device=device)
        # choose a subset of timesteps
        ts = torch.linspace(self.T-1, 0, steps, device=device).long()
        cvec = torch.full((n,), c, device=device, dtype=torch.long)

        for i in range(len(ts)-1):
            t = ts[i]
            t_prev = ts[i+1]
            eps_hat = self.eps_model(x, t.expand(n), cvec)
            a_t = self.alpha_bar[t]
            a_prev = self.alpha_bar[t_prev]
            # predicted x0
            x0_hat = (x - torch.sqrt(1-a_t) * eps_hat) / torch.sqrt(a_t)
            # direction to xt
            sigma = eta * torch.sqrt((1-a_prev)/(1-a_t) * (1 - a_t/a_prev))
            noise = sigma * torch.randn_like(x) if sigma > 0 else 0.0
            x = torch.sqrt(a_prev) * x0_hat + torch.sqrt(1-a_prev - sigma**2) * eps_hat + noise
        return x
