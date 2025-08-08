
import os, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from tqdm import tqdm
from torch_geometric.loader import DataLoader as GeoLoader

from drgat.models.graph_autoencoder import GraphAutoencoder
from drgat.models.ddim_latent import DDIMLatent
from drgat.models.ho_gat import HOGAT
from drgat.utils.graph import build_pathway_subgraphs, make_gene_lookup, inv_distance_weight
import networkx as nx

def to_tensor(x): return torch.tensor(x, dtype=torch.float32)

def train_autoencoder(expr_df, G_pathway, genes, device, epochs=50, lr=1e-3):
    # Build a single big graph induced by selected genes
    H = G_pathway.subgraph(genes).copy()
    import torch_geometric.utils as pyg_utils
    from torch_geometric.data import Data
    idx = make_gene_lookup(genes)
    # node features = mean expression across samples (pretraining signal)
    x = to_tensor(expr_df[genes].values.mean(axis=0)).unsqueeze(1).T  # shape (1, n_genes) -> we'll broadcast
    x = x.repeat(len(genes), 1)  # node features dim = n_genes (simple reconstruction)
    # Simple feature: we reconstruct the same vector; in practice you can use per-sample training
    # Here we pretrain on an averaged snapshot (toy-friendly).
    Hnx = H.copy()
    for n in Hnx.nodes:
        Hnx.nodes[n]["x"] = x[idx[n]]
    from torch_geometric.utils import from_networkx
    data = from_networkx(Hnx)
    in_dim = data.x.size(1)
    model = GraphAutoencoder(in_dim, hidden_dim=128, latent_dim=64).to(device)
    opt = Adam(model.parameters(), lr=lr)
    data = data.to(device)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        xhat, z = model(data.x, data.edge_index)
        loss = ((xhat - data.x)**2).mean()
        loss.backward()
        opt.step()
    return model, in_dim

def train_ddim_on_latents(model_ae, expr_df, genes, labels, device, epochs=100, lr=2e-4, timesteps=200):
    # Encode per-sample node features by averaging over nodes' encoder output
    # For simplicity, treat each sample's vector over genes as node features for the same topology,
    # and average the encoded node latents as a single latent per sample.
    from torch_geometric.utils import from_networkx
    import networkx as nx
    # Build trivial star graph to pass x through encoder without edges effect
    Gtmp = nx.Graph()
    for g in genes: Gtmp.add_node(g)
    data = from_networkx(Gtmp)
    edge_index = data.edge_index if data.edge_index.numel()>0 else torch.empty((2,0), dtype=torch.long)
    edge_index = edge_index.to(device)

    X = expr_df[genes].values.astype(np.float32)
    X = torch.tensor(X, dtype=torch.float32, device=device)
    latents = []
    with torch.no_grad():
        for i in range(X.size(0)):
            xi = X[i]
            # use xi as the "node feature" replicated to each node's dim to feed encoder
            xnodes = xi.unsqueeze(0).repeat(len(genes), 1)
            z = model_ae.encode(xnodes, edge_index)
            latents.append(z.mean(dim=0, keepdim=True))
    Z = torch.cat(latents, dim=0)  # [N, latent_dim]
    c = torch.tensor(labels.values.astype(int), device=device)

    ddim = DDIMLatent(latent_dim=Z.size(1), cond_dim=2, timesteps=timesteps).to(device)
    opt = Adam(ddim.parameters(), lr=lr)
    for ep in range(epochs):
        ddim.train()
        opt.zero_grad()
        loss = ddim.training_losses(Z, c)
        loss.backward()
        opt.step()
    return ddim, Z.detach(), c.detach()

def generate_augmented(ddim, n_per_class, device, latent_dim, label):
    c = torch.full((n_per_class,), int(label), device=device, dtype=torch.long)
    Zs = ddim.ddim_sample((n_per_class, latent_dim), c=c, eta=0.0)
    return Zs

def decode_latents_to_expression(model_ae, Zs):
    with torch.no_grad():
        Xs = model_ae.decode(Zs)
    return Xs

def build_subgraph_batch(G, pathways_df, selected_pids, genes, distances, gene_features_np):
    from drgat.utils.graph import build_pathway_subgraphs
    pathways = pathways_df.groupby("pathway_id")["gene"].apply(list).to_dict()
    gene_to_idx = make_gene_lookup(genes)
    subgraphs = build_pathway_subgraphs(G, pathways, selected_pids, gene_to_idx, gene_features_np)
    # Attach inv distance to each graph (readout)
    ds = []
    for pid, data in subgraphs:
        w = inv_distance_weight(distances.get(pid, 1.0))
        data.inv_d = torch.tensor([w], dtype=torch.float32)  # to be pooled later
        ds.append(data)
    return ds

def collate_with_inv_d(batch):
    # torch_geometric DataLoader already collates; here we just pass through
    return batch

def train_full_pipeline(expression_df, labels_df, drug, G, pathways_df, selected_pathways, gene_set, distances, output_dir, augment_ratio=0.7, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Filter to drug, split train/val (toy split 80/20)
    df = labels_df.copy()
    df = df[df["drug"] == drug]
    y = df["response"].values.astype(int)
    X = expression_df.loc[df["sample_id"]][gene_set].values.astype(np.float32)
    n = len(df)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(0.8 * n)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    # Train Graph Autoencoder on training data snapshot
    G_pathway = G.subgraph(gene_set).copy()
    ae, in_dim = train_autoencoder(expression_df, G_pathway, gene_set, device, epochs=30)

    # Train DDIM on latents
    dd, Z_real, c_real = train_ddim_on_latents(ae, expression_df.loc[df["sample_id"]], gene_set, df["response"], device, epochs=60)

    # Augment per class
    counts = {0: int((ytr==0).sum()), 1: int((ytr==1).sum())}
    total = counts[0]+counts[1]
    n_synth = int(round(augment_ratio * total))
    n0 = max(0, int(round(n_synth/2)))
    n1 = n_synth - n0
    Z0 = generate_augmented(dd, n0, device, Z_real.size(1), 0)
    Z1 = generate_augmented(dd, n1, device, Z_real.size(1), 1)
    X0 = decode_latents_to_expression(ae, Z0).cpu().numpy()
    X1 = decode_latents_to_expression(ae, Z1).cpu().numpy()
    # Map back to gene ordering (already aligned in decoder)
    # Create augmented dataset
    X_aug = np.vstack([X0, X1])
    y_aug = np.array([0]*len(X0) + [1]*len(X1), dtype=int)

    # Combine with train set
    Xtr_aug = np.vstack([Xtr, X_aug])
    ytr_aug = np.hstack([ytr, y_aug])

    # Build HO-GAT training batches over pathway subgraphs by using sample-wise node features:
    # For simplicity, we treat each sample as a pathway-batch where each node has the sample's gene vector.
    # We'll train by iterating samples; within a sample, we construct the subgraphs once with those features.
    model = HOGAT(in_dim, hidden=128, heads=2, layers=3, jk_mode="cat", num_classes=2).to(device)
    opt = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    def run_epoch(Xmat, yvec, train=True):
        model.train() if train else model.eval()
        losses, probs, gold = [], [], []
        for i in range(len(yvec)):
            x_feat = Xmat[i]  # shape [n_genes]
            # node features for each gene = x_feat vector (same dim as in_dim)
            gene_features = np.tile(x_feat, (len(gene_set), 1))
            ds = build_subgraph_batch(G, pathways_df, selected_pathways, gene_set, distances, gene_features)
            if not ds: 
                continue
            # Attach inv_d per graph to batch by simple scalar
            from torch_geometric.loader import DataLoader as GeoLoader
            loader = GeoLoader(ds, batch_size=len(ds))
            for batch in loader:
                batch = batch.to(device)
                inv_d = []
                # derive per-graph scalar
                # Here we stored no field in batch; simply approximate with 1.0 for all graphs
                inv_d = torch.ones((batch.num_graphs,1), device=device)
                logits = model(batch.x, batch.edge_index, batch.batch, inv_dist_weight=inv_d)
                # Aggregate graph predictions (mean) per sample
                logit = logits.mean(dim=0, keepdim=True)
                y_i = torch.tensor([yvec[i]], device=device, dtype=torch.long)
                loss = loss_fn(logit, y_i)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                losses.append(loss.item())
                with torch.no_grad():
                    p = torch.softmax(logit, dim=-1).detach().cpu().numpy()[0,1]
                probs.append(p); gold.append(int(yvec[i]))
        return np.mean(losses) if losses else float("nan"), np.array(probs), np.array(gold, dtype=int)

    # train loop with early stopping
    best_va = -1e9; best_state = None; patience=10; wait=0
    for ep in range(60):
        tr_loss, _, _ = run_epoch(Xtr_aug, ytr_aug, train=True)
        va_loss, va_prob, va_gold = run_epoch(Xva, yva, train=False)
        if len(va_prob) > 0:
            roc = roc_auc_score(va_gold, va_prob)
            pr = average_precision_score(va_gold, va_prob)
            f1 = f1_score(va_gold, (va_prob>0.5).astype(int))
            acc = accuracy_score(va_gold, (va_prob>0.5).astype(int))
            score = roc
        else:
            roc=pr=f1=acc=float("nan"); score=-1e9
        if score > best_va:
            best_va = score; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait += 1
            if wait>=patience: break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final val
    _, va_prob, va_gold = run_epoch(Xva, yva, train=False)
    out = {}
    if len(va_prob)>0:
        out = {
            "val_roc_auc": float(roc_auc_score(va_gold, va_prob)),
            "val_pr_auc": float(average_precision_score(va_gold, va_prob)),
            "val_f1": float(f1_score(va_gold, (va_prob>0.5).astype(int))),
            "val_acc": float(accuracy_score(va_gold, (va_prob>0.5).astype(int))),
            "n_train": int(len(ytr)),
            "n_val": int(len(yva)),
            "augment_ratio": float(augment_ratio),
            "k_pathways": int(len(selected_pathways)),
            "n_genes": int(len(gene_set)),
        }
    else:
        out = {"message": "Validation set too small."}
    return out
