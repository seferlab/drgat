"""Synthetic-vs-real evaluation metrics used for Table 6.

The attached manuscript's *Table 6* compares generation performance of
different models (CVAE/CTGAN/TVAE/CopulaGAN vs DRGAT) using four metrics:
KLD, Pairwise Difference (PD), Log-Cluster, and Cosine Similarity.

The original paper cites Goncalves et al. (2020) as inspiration for these
metrics. Implementations in the wild vary; this module provides a **robust,
deterministic and dependency-light** implementation suitable for reproducing
the table generation pipeline.

Notes
-----
* All metrics operate on *expression matrices* with shape (n_samples, n_genes).
* For stability and speed, PD uses random subsampling.
* Log-Cluster is implemented as the log of the Jensen–Shannon divergence
  between cluster membership distributions obtained from KMeans on a PCA
  projection. This yields negative values (as in the paper) when the
  divergence is < 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _safe_hist_pmf(x: np.ndarray, bins: int = 20, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pmf, bin_edges) for a 1D array."""
    hist, edges = np.histogram(x, bins=bins, density=False)
    hist = hist.astype(np.float64)
    pmf = hist + eps
    pmf = pmf / pmf.sum()
    return pmf, edges


def kld_marginal(real: np.ndarray, synth: np.ndarray, bins: int = 20, eps: float = 1e-12) -> float:
    """Average marginal KL(real || synth) across features.

    Both distributions are estimated via histograms using the *same* bin edges
    defined from real data per feature.
    """
    real = np.asarray(real)
    synth = np.asarray(synth)
    assert real.ndim == 2 and synth.ndim == 2
    d = real.shape[1]
    kls = []
    for j in range(d):
        p, edges = _safe_hist_pmf(real[:, j], bins=bins, eps=eps)
        hist_q, _ = np.histogram(synth[:, j], bins=edges, density=False)
        q = hist_q.astype(np.float64) + eps
        q = q / q.sum()
        kls.append(float(np.sum(p * (np.log(p) - np.log(q)))))
    return float(np.mean(kls))


def cosine_similarity_mean(real: np.ndarray, synth: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between mean vectors of real and synthetic."""
    mr = np.mean(real, axis=0)
    ms = np.mean(synth, axis=0)
    num = float(np.dot(mr, ms))
    den = float(np.linalg.norm(mr) * np.linalg.norm(ms) + eps)
    return num / den


def pairwise_difference(real: np.ndarray, synth: np.ndarray, max_pairs: int = 2048, seed: int = 42) -> float:
    """Mean Euclidean distance between randomly paired real and synthetic samples."""
    rng = np.random.default_rng(int(seed))
    n = min(real.shape[0], synth.shape[0])
    if n <= 0:
        return float("nan")
    k = min(int(max_pairs), int(n))
    idx_r = rng.choice(n, size=k, replace=False)
    idx_s = rng.choice(n, size=k, replace=False)
    dr = real[idx_r]
    ds = synth[idx_s]
    dist = np.linalg.norm(dr - ds, axis=1)
    return float(np.mean(dist))


def _pca_project(X: np.ndarray, n_components: int = 50) -> np.ndarray:
    """Simple PCA via SVD (no sklearn dependency required)."""
    X = X.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    # economy SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    return (U[:, :k] * S[:k]).astype(np.float32)


def _kmeans(X: np.ndarray, k: int, seed: int = 42, n_iter: int = 50) -> np.ndarray:
    """Very small KMeans implementation (Lloyd's algorithm)."""
    rng = np.random.default_rng(int(seed))
    n = X.shape[0]
    # init centers from random points
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(int(n_iter)):
        # assign
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # update
        for ci in range(k):
            mask = labels == ci
            if mask.any():
                centers[ci] = X[mask].mean(axis=0)
    return labels


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    return float(0.5 * (kl_pm + kl_qm))


def log_cluster(real: np.ndarray, synth: np.ndarray, k: int = 10, seed: int = 42) -> float:
    """Log-Cluster metric (lower is better).

    Steps:
      1) PCA project concatenated data.
      2) KMeans on concatenated projected data.
      3) Compare real vs synth cluster membership distributions via JS divergence.
      4) Return log(JS + eps).
    """
    X = np.vstack([real, synth])
    Z = _pca_project(X, n_components=50)
    labels = _kmeans(Z, k=int(k), seed=int(seed))
    n_r = real.shape[0]
    lr = labels[:n_r]
    ls = labels[n_r:]
    pr = np.bincount(lr, minlength=k).astype(np.float64)
    ps = np.bincount(ls, minlength=k).astype(np.float64)
    js = _js_divergence(pr, ps)
    return float(np.log(js + 1e-12))


@dataclass
class SynthMetricResult:
    kld: float
    pd: float
    log_cluster: float
    cosine_sim: float


def compute_all(real: np.ndarray, synth: np.ndarray, seed: int = 42) -> SynthMetricResult:
    """Compute all four metrics."""
    return SynthMetricResult(
        kld=kld_marginal(real, synth),
        pd=pairwise_difference(real, synth, seed=seed),
        log_cluster=log_cluster(real, synth, seed=seed),
        cosine_sim=cosine_similarity_mean(real, synth),
    )
