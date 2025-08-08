
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

def _avg_shortest_path_to_targets(G, pathway_genes, target_genes):
    # Use average of min distances from each target to the pathway gene set
    dists = []
    target_genes = [t for t in target_genes if t in G]
    if not target_genes or len(pathway_genes)==0:
        return np.inf
    # precompute shortest paths once
    sp = dict(nx.all_pairs_shortest_path_length(G, cutoff=5))
    for t in target_genes:
        if t not in sp: 
            continue
        # min distance from t to any gene in pathway_genes
        m = np.inf
        for s in pathway_genes:
            if s in sp[t]:
                m = min(m, sp[t][s])
        if np.isfinite(m):
            dists.append(m)
    return float(np.mean(dists)) if dists else np.inf

class PathwaySelector:
    def __init__(self, G, pathways_df: pd.DataFrame):
        self.G = G
        self.pathways = pathways_df.groupby("pathway_id")["gene"].apply(list).to_dict()

    def select_for_drug(self, drug: str, target_genes, k_ratio=0.05, n_boot=200, seed=42):
        rng = np.random.default_rng(seed)
        # compute observed distances
        obs = {}
        for pid, genes in self.pathways.items():
            genes = [g for g in genes if g in self.G]
            d = _avg_shortest_path_to_targets(self.G, genes, target_genes or [])
            obs[pid] = d
        # bootstrap null: degree/size-matched random sets
        # approximate: sample random nodes with same size
        all_genes = [n for n in self.G.nodes]
        ref_stats = {}
        for pid, genes in tqdm(self.pathways.items(), desc="Bootstrap null"):
            k = len([g for g in genes if g in self.G])
            if k == 0:
                ref_stats[pid] = (np.nan, np.nan)
                continue
            vals = []
            for _ in range(n_boot):
                S = rng.choice(all_genes, size=k, replace=False)
                vals.append(_avg_shortest_path_to_targets(self.G, S, target_genes or []))
            mu = np.nanmean(vals)
            sd = np.nanstd(vals) + 1e-8
            ref_stats[pid] = (mu, sd)
        # z-scores: smaller distance => more proximal (invert sign)
        z = {}
        for pid, d in obs.items():
            mu, sd = ref_stats[pid]
            if np.isfinite(d) and np.isfinite(mu):
                z[pid] = -(d - mu) / sd
            else:
                z[pid] = -np.inf
        # select top-K
        K = max(1, int(round(k_ratio * len(self.pathways))))
        selected = [pid for pid, _ in sorted(z.items(), key=lambda kv: kv[1], reverse=True)[:K]]
        gene_set = sorted({g for pid in selected for g in self.pathways[pid] if g in self.G})
        # distances for readout weighting: inverse distance proxy per pathway (avoid div0)
        distances = {pid: max(1.0, obs[pid]) for pid in selected}
        return selected, gene_set, distances
