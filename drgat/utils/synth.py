"""Lightweight tabular synthetic data generators used for Table 5.

The paper compares DRGAT's diffusion-based augmentation against several
tabular generators (CVAE/CTGAN/TVAE/CopulaGAN). The original baselines
use dedicated libraries (e.g., SDV). To keep this repository self-contained
and runnable with the existing requirements, we provide *minimal* PyTorch/
NumPy implementations that follow the same high-level idea:

- CVAE: conditional VAE on expression vectors
- CTGAN: simple conditional GAN (MLP generator/discriminator)
- TVAE: VAE trained per-class (tabular VAE)
- CopulaGAN: Gaussian-copula sampling per-class

These implementations are intended to reproduce the *table-generation
pipeline* in a robust way and may not match SDV's exact hyper-parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange

from drgat.models.vae_tabular import TabularVAE, vae_loss


def _counts_for_ratio(y: np.ndarray, augment_ratio: float) -> Tuple[int, int]:
    total = int(y.shape[0])
    n_synth = int(round(float(augment_ratio) * total))
    n0 = int(round(n_synth / 2))
    n1 = int(n_synth - n0)
    return n0, n1


# -------------------- CVAE --------------------


class _CVAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 64, hidden_dim: int = 256, n_classes: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.y_emb = nn.Embedding(n_classes, 16)

        self.enc = nn.Sequential(
            nn.Linear(in_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        yv = self.y_emb(y)
        h = self.enc(torch.cat([x, yv], dim=-1))
        return self.mu(h), self.logvar(h)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        yv = self.y_emb(y)
        return self.dec(torch.cat([z, yv], dim=-1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparam(mu, logvar)
        xhat = self.decode(z, y)
        return xhat, mu, logvar


def _cvae_loss(xhat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> torch.Tensor:
    recon = ((xhat - x) ** 2).mean()
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld


def fit_and_sample_cvae(
    X: np.ndarray,
    y: np.ndarray,
    augment_ratio: float,
    seed: int,
    device: torch.device,
    latent_dim: int = 64,
    epochs: int = 120,
    lr: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    if augment_ratio <= 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=int)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    x = torch.tensor(X.astype(np.float32), device=device)
    yt = torch.tensor(y.astype(int), device=device, dtype=torch.long)
    model = _CVAE(in_dim=x.shape[1], latent_dim=latent_dim).to(device)
    opt = Adam(model.parameters(), lr=lr)

    # KL annealing (simple linear schedule)
    for t in trange(epochs, desc="CVAE", leave=False):
        beta = min(1.0, (t + 1) / max(1, epochs // 2))
        model.train()
        opt.zero_grad(set_to_none=True)
        xhat, mu, logvar = model(x, yt)
        loss = _cvae_loss(xhat, x, mu, logvar, beta=beta)
        loss.backward()
        opt.step()

    n0, n1 = _counts_for_ratio(y, augment_ratio)
    if n0 + n1 <= 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=int)

    model.eval()
    with torch.no_grad():
        z0 = torch.randn((n0, latent_dim), device=device)
        z1 = torch.randn((n1, latent_dim), device=device)
        y0 = torch.zeros((n0,), device=device, dtype=torch.long)
        y1 = torch.ones((n1,), device=device, dtype=torch.long)
        x0 = model.decode(z0, y0)
        x1 = model.decode(z1, y1)
        xs = torch.cat([x0, x1], dim=0).detach().cpu().numpy().astype(np.float32)
        ys = np.array([0] * n0 + [1] * n1, dtype=int)
    return xs, ys


# -------------------- CTGAN (simple conditional GAN) --------------------


class _CondMLPGen(nn.Module):
    def __init__(self, noise_dim: int, out_dim: int, hidden_dim: int = 256, n_classes: int = 2):
        super().__init__()
        self.y_emb = nn.Embedding(n_classes, 16)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        yv = self.y_emb(y)
        return self.net(torch.cat([z, yv], dim=-1))


class _CondMLPDisc(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, n_classes: int = 2):
        super().__init__()
        self.y_emb = nn.Embedding(n_classes, 16)
        self.net = nn.Sequential(
            nn.Linear(in_dim + 16, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        yv = self.y_emb(y)
        return self.net(torch.cat([x, yv], dim=-1))


def fit_and_sample_ctgan(
    X: np.ndarray,
    y: np.ndarray,
    augment_ratio: float,
    seed: int,
    device: torch.device,
    noise_dim: int = 64,
    steps: int = 1500,
    batch_size: int = 128,
    lr: float = 2e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    if augment_ratio <= 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=int)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    x = torch.tensor(X.astype(np.float32), device=device)
    yt = torch.tensor(y.astype(int), device=device, dtype=torch.long)
    gen = _CondMLPGen(noise_dim=noise_dim, out_dim=x.shape[1]).to(device)
    disc = _CondMLPDisc(in_dim=x.shape[1]).to(device)
    opt_g = Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = Adam(disc.parameters(), lr=lr, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss()

    n = x.shape[0]
    rng = np.random.default_rng(int(seed))

    for _ in trange(steps, desc="CTGAN", leave=False):
        # minibatch
        idx = rng.integers(0, n, size=(min(batch_size, n),))
        xb = x[idx]
        yb = yt[idx]

        # train discriminator
        z = torch.randn((xb.size(0), noise_dim), device=device)
        x_fake = gen(z, yb).detach()
        d_real = disc(xb, yb)
        d_fake = disc(x_fake, yb)
        loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        # train generator
        z = torch.randn((xb.size(0), noise_dim), device=device)
        x_fake = gen(z, yb)
        d_fake = disc(x_fake, yb)
        loss_g = bce(d_fake, torch.ones_like(d_fake))
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

    n0, n1 = _counts_for_ratio(y, augment_ratio)
    gen.eval()
    with torch.no_grad():
        z0 = torch.randn((n0, noise_dim), device=device)
        z1 = torch.randn((n1, noise_dim), device=device)
        y0 = torch.zeros((n0,), device=device, dtype=torch.long)
        y1 = torch.ones((n1,), device=device, dtype=torch.long)
        x0 = gen(z0, y0)
        x1 = gen(z1, y1)
        xs = torch.cat([x0, x1], dim=0).detach().cpu().numpy().astype(np.float32)
        ys = np.array([0] * n0 + [1] * n1, dtype=int)
    return xs, ys


# -------------------- TVAE (per-class VAE) --------------------


def fit_and_sample_tvae(
    X: np.ndarray,
    y: np.ndarray,
    augment_ratio: float,
    seed: int,
    device: torch.device,
    latent_dim: int = 64,
    epochs: int = 120,
    lr: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    if augment_ratio <= 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=int)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    n0, n1 = _counts_for_ratio(y, augment_ratio)
    xs = []
    ys = []
    for cls, n_s in [(0, n0), (1, n1)]:
        if n_s <= 0:
            continue
        Xc = X[y == cls]
        if Xc.shape[0] < 4:
            continue
        xtr = torch.tensor(Xc.astype(np.float32), device=device)
        vae = TabularVAE(in_dim=xtr.size(1), latent_dim=latent_dim, hidden_dim=256).to(device)
        opt = Adam(vae.parameters(), lr=lr)
        vae.train()
        for _ in trange(epochs, desc=f"TVAE(c={cls})", leave=False):
            opt.zero_grad(set_to_none=True)
            xhat, mu, logvar = vae(xtr)
            loss, _, _ = vae_loss(xhat, xtr, mu, logvar, beta=1.0)
            loss.backward()
            opt.step()
        vae.eval()
        with torch.no_grad():
            z = torch.randn((n_s, latent_dim), device=device)
            xg = vae.decode(z).detach().cpu().numpy().astype(np.float32)
        xs.append(xg)
        ys.append(np.full((xg.shape[0],), cls, dtype=int))
    if not xs:
        return np.zeros((0, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=int)
    return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(int)


# -------------------- CopulaGAN (Gaussian copula, per-class) --------------------


@dataclass
class _CopulaModel:
    mu: np.ndarray
    cov: np.ndarray
    quantiles: np.ndarray  # [d, q]
    qgrid: np.ndarray      # [q]


def _fit_copula(X: np.ndarray, q: int = 512, eps: float = 1e-6) -> _CopulaModel:
    # empirical CDF -> Gaussianize
    n, d = X.shape
    ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float64)
    u = (ranks + 1.0) / (n + 2.0)
    # inverse normal
    z = np.sqrt(2) * erfinv(2 * u - 1)
    mu = z.mean(axis=0)
    cov = np.cov(z, rowvar=False) + eps * np.eye(d)

    # store per-feature quantile function for inverse
    qgrid = np.linspace(0.0, 1.0, q)
    quantiles = np.quantile(X, qgrid, axis=0).T  # [d, q]
    return _CopulaModel(mu=mu, cov=cov, quantiles=quantiles, qgrid=qgrid)


def _sample_copula(model: _CopulaModel, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    z = rng.multivariate_normal(mean=model.mu, cov=model.cov, size=n)
    # normal CDF
    u = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    u = np.clip(u, 0.0, 1.0)
    # inverse via quantile interpolation
    d = model.quantiles.shape[0]
    Xs = np.zeros((n, d), dtype=np.float32)
    for j in range(d):
        Xs[:, j] = np.interp(u[:, j], model.qgrid, model.quantiles[j]).astype(np.float32)
    return Xs


def fit_and_sample_copula(
    X: np.ndarray,
    y: np.ndarray,
    augment_ratio: float,
    seed: int,
    q: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    if augment_ratio <= 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=int)
    n0, n1 = _counts_for_ratio(y, augment_ratio)
    xs, ys = [], []
    for cls, n_s in [(0, n0), (1, n1)]:
        if n_s <= 0:
            continue
        Xc = X[y == cls]
        if Xc.shape[0] < 10:
            continue
        m = _fit_copula(Xc, q=q)
        xs.append(_sample_copula(m, n_s, seed + 1000 + cls))
        ys.append(np.full((n_s,), cls, dtype=int))
    if not xs:
        return np.zeros((0, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=int)
    return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(int)


# -------------------- small numeric helpers (no scipy dependency) --------------------


def erf(x: np.ndarray) -> np.ndarray:
    # Abramowitz-Stegun approximation
    # https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
    sign = np.sign(x)
    x = np.abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def erfinv(x: np.ndarray) -> np.ndarray:
    # Approximate inverse error function (Winitzki)
    # https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    a = 0.147
    x = np.clip(x, -0.999999, 0.999999)
    ln = np.log(1.0 - x * x)
    first = 2.0 / (np.pi * a) + ln / 2.0
    second = ln / a
    return np.sign(x) * np.sqrt(np.sqrt(first * first - second) - first)
