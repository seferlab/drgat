
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import json
import numpy as np
import pandas as pd
import torch
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from torch.optim import Adam
from tqdm import trange

from drgat.models.graph_autoencoder import GraphAutoencoder
from drgat.models.ddim_latent import DDIMLatent
from drgat.models.ho_gat import HOGAT
from drgat.models.gat_simple import SimpleGAT
from drgat.models.vae_tabular import TabularVAE, vae_loss
from drgat.utils.synth import (
    fit_and_sample_copula,
    fit_and_sample_ctgan,
    fit_and_sample_cvae,
    fit_and_sample_tvae,
)
from drgat.utils.graph import build_pathway_subgraphs, make_gene_lookup, inv_distance_weight


def _device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_graph_autoencoder(
    train_expr: pd.DataFrame,
    G_pathway: nx.Graph,
    genes: list[str],
    device: torch.device,
    hidden_dim: int = 128,
    latent_dim: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
) -> Tuple[GraphAutoencoder, torch.Tensor]:
    """Graph autoencoder pretraining.

    Simplified but consistent with the repo's existing design:
    - each node gets the *full sample gene vector* as its feature
    - we optimize reconstruction error on node-features
    """
    from torch_geometric.utils import from_networkx

    pyg = from_networkx(G_pathway)
    edge_index = pyg.edge_index.to(device)

    in_dim = len(genes)
    ae = GraphAutoencoder(in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    opt = Adam(ae.parameters(), lr=lr)

    X = torch.tensor(train_expr[genes].values.astype(np.float32), device=device)

    for _ in trange(epochs, desc="AE", leave=False):
        ae.train()
        for i in range(X.size(0)):
            xi = X[i]
            xnodes = xi.unsqueeze(0).repeat(len(genes), 1)  # [n_genes, in_dim]
            opt.zero_grad(set_to_none=True)
            xhat, _ = ae(xnodes, edge_index)
            loss = ((xhat - xnodes) ** 2).mean()
            loss.backward()
            opt.step()
    return ae, edge_index


@torch.no_grad()
def encode_latents(ae: GraphAutoencoder, edge_index: torch.Tensor, expr: pd.DataFrame, genes: list[str]) -> torch.Tensor:
    ae.eval()
    X = torch.tensor(expr[genes].values.astype(np.float32), device=edge_index.device)
    Z = []
    for i in range(X.size(0)):
        xi = X[i]
        xnodes = xi.unsqueeze(0).repeat(len(genes), 1)
        _, z_nodes = ae(xnodes, edge_index)
        Z.append(z_nodes.mean(dim=0, keepdim=True))
    return torch.cat(Z, dim=0)


def train_ddim(Z: torch.Tensor, y: np.ndarray, device: torch.device, timesteps: int = 200, epochs: int = 200, lr: float = 2e-4) -> DDIMLatent:
    ddim = DDIMLatent(latent_dim=Z.size(1), cond_dim=2, timesteps=timesteps).to(device)
    opt = Adam(ddim.parameters(), lr=lr)
    y_t = torch.tensor(y.astype(int), device=device, dtype=torch.long)
    for _ in trange(epochs, desc="DDIM", leave=False):
        ddim.train()
        opt.zero_grad(set_to_none=True)
        loss = ddim.training_losses(Z, y_t)
        loss.backward()
        opt.step()
    return ddim


@torch.no_grad()
def sample_ddim(ddim: DDIMLatent, n: int, latent_dim: int, cls: int, device: torch.device) -> torch.Tensor:
    c = torch.full((n,), int(cls), device=device, dtype=torch.long)
    return ddim.ddim_sample((n, latent_dim), c=c, eta=0.0)


def build_subgraphs_for_sample(
    G: nx.Graph,
    pathways_df: pd.DataFrame,
    selected_pids: list[str],
    genes: list[str],
    distances: Dict[str, float],
    gene_features_np: np.ndarray,
):
    pathways = pathways_df.groupby("pathway_id")["gene"].apply(list).to_dict()
    gene_to_idx = make_gene_lookup(genes)
    subgraphs = build_pathway_subgraphs(G, pathways, selected_pids, gene_to_idx, gene_features_np)
    ds = []
    for pid, data in subgraphs:
        w = inv_distance_weight(distances.get(pid, 1.0))
        data.inv_d = torch.tensor([w], dtype=torch.float32)
        ds.append(data)
    return ds


def train_predictor(
    X: np.ndarray,
    y: np.ndarray,
    G: nx.Graph,
    pathways_df: pd.DataFrame,
    selected_pathways: list[str],
    genes: list[str],
    distances: Dict[str, float],
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-3,
    predictor_type: str = "hogat",
    patience: int = 10,
) -> HOGAT:
    from torch_geometric.loader import DataLoader as GeoLoader
    loss_fn = torch.nn.CrossEntropyLoss()

    in_dim = X.shape[1]
    model = (HOGAT if predictor_type == "hogat" else SimpleGAT)(
        in_dim, hidden=128, heads=2, layers=3, jk_mode="cat", num_classes=2
    ).to(device)
    opt = Adam(model.parameters(), lr=lr)

    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(1337)
    rng.shuffle(idx)
    n_tr = int(0.8 * n)
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]

    def run(idxs, train: bool):
        model.train(train)
        probs, gold, losses = [], [], []
        for i in idxs:
            x_feat = X[i]
            gene_features = np.tile(x_feat, (len(genes), 1))
            ds = build_subgraphs_for_sample(G, pathways_df, selected_pathways, genes, distances, gene_features)
            if not ds:
                continue
            loader = GeoLoader(ds, batch_size=len(ds), shuffle=False)
            for batch in loader:
                batch = batch.to(device)
                inv_d = batch.inv_d.to(device).view(-1, 1) if hasattr(batch, "inv_d") else torch.ones((batch.num_graphs, 1), device=device)
                logits_graphs = model(batch.x, batch.edge_index, batch.batch, inv_dist_weight=inv_d)
                logit = logits_graphs.mean(dim=0, keepdim=True)
                y_i = torch.tensor([int(y[i])], device=device, dtype=torch.long)
                loss = loss_fn(logit, y_i)
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                losses.append(float(loss.item()))
                p = torch.softmax(logit, dim=-1)[:, 1].detach().cpu().numpy()[0]
                probs.append(p); gold.append(int(y[i]))
        if not probs:
            return float("nan"), np.array([]), np.array([])
        return float(np.mean(losses)), np.array(probs), np.array(gold, dtype=int)

    best_auc = -1.0
    best_state = None
    bad = 0
    for _ in trange(epochs, desc="Predictor", leave=False):
        run(tr_idx, train=True)
        _, p, g = run(va_idx, train=False)
        if len(p) >= 2 and len(np.unique(g)) >= 2:
            auc = roc_auc_score(g, p)
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model: HOGAT, X: np.ndarray, G: nx.Graph, pathways_df: pd.DataFrame, selected_pathways: list[str], genes: list[str], distances: Dict[str, float], device: torch.device) -> np.ndarray:
    from torch_geometric.loader import DataLoader as GeoLoader
    model.eval()
    probs = []
    for i in range(X.shape[0]):
        x_feat = X[i]
        gene_features = np.tile(x_feat, (len(genes), 1))
        ds = build_subgraphs_for_sample(G, pathways_df, selected_pathways, genes, distances, gene_features)
        if not ds:
            probs.append(np.nan)
            continue
        loader = GeoLoader(ds, batch_size=len(ds), shuffle=False)
        for batch in loader:
            batch = batch.to(device)
            inv_d = batch.inv_d.to(device).view(-1, 1) if hasattr(batch, "inv_d") else torch.ones((batch.num_graphs, 1), device=device)
            logits_graphs = model(batch.x, batch.edge_index, batch.batch, inv_dist_weight=inv_d)
            logit = logits_graphs.mean(dim=0, keepdim=True)
            p = torch.softmax(logit, dim=-1)[:, 1].detach().cpu().numpy()[0]
            probs.append(p)
    return np.array(probs, dtype=float)


def train_and_eval_drgat(
    train_expr: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_expr: pd.DataFrame,
    test_labels: pd.DataFrame,
    drug: str,
    G: nx.Graph,
    pathways_df: pd.DataFrame,
    selected_pathways: list[str],
    gene_set: list[str],
    distances: Dict[str, float],
    augment_ratio: float,
    seed: int = 42,
    device: str = "auto",
    aug_method: str = "latent_ae",
    predictor_type: str = "hogat",
    out_dir: Path | None = None,
) -> Dict[str, Any]:
    if out_dir is None:
        raise ValueError("train_and_eval_drgat requires out_dir.")
    out_dir.mkdir(parents=True, exist_ok=True)
    #torch.manual_seed(seed)
    #np.random.seed(seed)

    dev = _device(device)
    genes = list(gene_set)

    # align labels to expr
    y_train = train_labels.set_index("sample_id").loc[train_expr.index]["response"].values.astype(int)
    y_test = test_labels.set_index("sample_id").loc[test_expr.index]["response"].values.astype(int)

    # induce topology
    G_pathway = G.subgraph(genes).copy()

    # --- Augmentation backbone (Tables 1/4/5) ---
    # aug_method:
    #   - "latent_ae": latent diffusion on (graph) autoencoder latent space (default; paper's DRGAT)
    #   - "diff_direct": diffusion directly in expression space (no AE)
    #   - "latent_vae": latent diffusion on a tabular VAE latent space
    #   - "cvae": CVAE-based synthetic augmentation (Table 5)
    #   - "ctgan": lightweight conditional GAN (Table 5)
    #   - "tvae": tabular VAE augmentation (Table 5)
    #   - "copulagan": Gaussian-copula augmentation (Table 5)
    X_train = train_expr[genes].values.astype(np.float32)

    if aug_method == "latent_ae":
        ae, edge_index = train_graph_autoencoder(train_expr, G_pathway, genes, dev, epochs=int(60 if augment_ratio > 0 else 40))
        Z_train = encode_latents(ae, edge_index, train_expr, genes)

        if augment_ratio > 0:
            ddim = train_ddim(Z_train.detach(), y_train, dev, epochs=200)
            counts = {0: int((y_train == 0).sum()), 1: int((y_train == 1).sum())}
            total = counts[0] + counts[1]
            n_synth = int(round(augment_ratio * total))
            n0 = int(round(n_synth / 2))
            n1 = n_synth - n0
            Z0 = sample_ddim(ddim, n0, Z_train.size(1), 0, dev)
            Z1 = sample_ddim(ddim, n1, Z_train.size(1), 1, dev)
            Zs = torch.cat([Z0, Z1], dim=0)
            ys = np.array([0] * n0 + [1] * n1, dtype=int)

            ae.eval()
            X_synth = ae.decode(Zs).detach().cpu().numpy().astype(np.float32)
            X_train = np.vstack([X_train, X_synth])
            y_train = np.concatenate([y_train, ys], axis=0)

    elif aug_method == "diff_direct":
        # Treat expression vectors as "latents" for diffusion.
        Z_train = torch.tensor(X_train, device=dev, dtype=torch.float32)
        if augment_ratio > 0:
            ddim = train_ddim(Z_train.detach(), y_train, dev, epochs=200)
            counts = {0: int((y_train == 0).sum()), 1: int((y_train == 1).sum())}
            total = counts[0] + counts[1]
            n_synth = int(round(augment_ratio * total))
            n0 = int(round(n_synth / 2))
            n1 = n_synth - n0
            X0 = sample_ddim(ddim, n0, Z_train.size(1), 0, dev).detach().cpu().numpy().astype(np.float32)
            X1 = sample_ddim(ddim, n1, Z_train.size(1), 1, dev).detach().cpu().numpy().astype(np.float32)
            X_synth = np.vstack([X0, X1])
            ys = np.array([0] * n0 + [1] * n1, dtype=int)
            X_train = np.vstack([X_train, X_synth])
            y_train = np.concatenate([y_train, ys], axis=0)

    elif aug_method == "latent_vae":
        xtr = torch.tensor(X_train, device=dev, dtype=torch.float32)
        vae = TabularVAE(in_dim=xtr.size(1), latent_dim=64, hidden_dim=256).to(dev)
        opt = Adam(vae.parameters(), lr=1e-3)

        vae.train()
        for _ in trange(int(80 if augment_ratio > 0 else 50), desc="VAE", leave=False):
            opt.zero_grad(set_to_none=True)
            xhat, mu, logvar = vae(xtr)
            loss, _, _ = vae_loss(xhat, xtr, mu, logvar, beta=1.0)
            loss.backward()
            opt.step()

        with torch.no_grad():
            vae.eval()
            mu, logvar = vae.encode(xtr)
            ztr = vae.reparameterize(mu, logvar)

        if augment_ratio > 0:
            ddim = train_ddim(ztr.detach(), y_train, dev, epochs=200)
            counts = {0: int((y_train == 0).sum()), 1: int((y_train == 1).sum())}
            total = counts[0] + counts[1]
            n_synth = int(round(augment_ratio * total))
            n0 = int(round(n_synth / 2))
            n1 = n_synth - n0
            Z0 = sample_ddim(ddim, n0, ztr.size(1), 0, dev)
            Z1 = sample_ddim(ddim, n1, ztr.size(1), 1, dev)
            Zs = torch.cat([Z0, Z1], dim=0)
            ys = np.array([0] * n0 + [1] * n1, dtype=int)

            with torch.no_grad():
                X_synth = vae.decode(Zs).detach().cpu().numpy().astype(np.float32)
            X_train = np.vstack([X_train, X_synth])
            y_train = np.concatenate([y_train, ys], axis=0)

    elif aug_method in {"cvae", "ctgan", "tvae", "copulagan"}:
        if augment_ratio > 0:
            if aug_method == "cvae":
                X_synth, y_synth = fit_and_sample_cvae(X_train, y_train, augment_ratio, seed=seed, device=dev)
            elif aug_method == "ctgan":
                X_synth, y_synth = fit_and_sample_ctgan(X_train, y_train, augment_ratio, seed=seed, device=dev)
            elif aug_method == "tvae":
                X_synth, y_synth = fit_and_sample_tvae(X_train, y_train, augment_ratio, seed=seed, device=dev)
            else:
                X_synth, y_synth = fit_and_sample_copula(X_train, y_train, augment_ratio, seed=seed)

            if X_synth.size > 0:
                X_train = np.vstack([X_train, X_synth.astype(np.float32)])
                y_train = np.concatenate([y_train, y_synth.astype(int)], axis=0)

    else:
        raise ValueError(
            f"Unknown aug_method: {aug_method}. Expected one of: "
            "latent_ae, diff_direct, latent_vae, cvae, ctgan, tvae, copulagan."
        )

    model = train_predictor(
        X_train,
        y_train,
        G,
        pathways_df,
        selected_pathways,
        genes,
        distances,
        dev,
        predictor_type=predictor_type,
    )

    X_test = test_expr[genes].values.astype(np.float32)
    prob = predict(model, X_test, G, pathways_df, selected_pathways, genes, distances, dev)
    mask = np.isfinite(prob)
    prob = prob[mask]
    y_test2 = y_test[mask]

    out: Dict[str, Any] = {"drug": drug, "augment_ratio": float(augment_ratio), "n_train": int(len(y_train)), "n_test": int(len(y_test2))}
    if len(prob) >= 2 and len(np.unique(y_test2)) >= 2:
        out.update({
            "test_roc_auc": float(roc_auc_score(y_test2, prob)),
            "test_pr_auc": float(average_precision_score(y_test2, prob)),
            "test_f1": float(f1_score(y_test2, (prob > 0.5).astype(int))),
            "test_acc": float(accuracy_score(y_test2, (prob > 0.5).astype(int))),
        })
    else:
        out.update({"test_roc_auc": float("nan"), "test_pr_auc": float("nan"), "test_f1": float("nan"), "test_acc": float("nan")})

    pd.DataFrame({"sample_id": test_expr.index[mask].tolist(), "y": y_test2.tolist(), "p": prob.tolist()}).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    return out
