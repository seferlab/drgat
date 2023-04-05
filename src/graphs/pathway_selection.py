from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import networkx as nx

@dataclass
class PathwayScore:
    pathway_id: str
    d_closest: float
    z: float

def dclosest(G_ppi: nx.Graph, pathway_genes: Iterable[str], target_genes: Iterable[str]) -> float:
    """Average over target genes of min shortest path to any pathway gene.

    Matches the definition in Eq.(5) of the paper.
    """
    pathway = [g for g in pathway_genes if g in G_ppi]
    targets = [g for g in target_genes if g in G_ppi]
    if not pathway or not targets:
        return float("inf")

    # Precompute shortest paths from pathway nodes to speed up.
    # For large graphs, consider multi-source Dijkstra / BFS (unweighted) per target.
    lengths = nx.multi_source_dijkstra_path_length(G_ppi, pathway, weight=None)
    vals = []
    for t in targets:
        vals.append(lengths.get(t, float("inf")))
    return float(np.mean(vals))

def degree_matched_bootstrap_genes(G_ppi: nx.Graph, target_genes: list[str], n: int, bins: int = 20, rng=None) -> list[str]:
    """Sample genes with approximately matched degree distribution to target_genes.

    The paper describes bootstrapping random gene sets with matched degree+size.
    This function bins degrees and samples within bins.
    """
    rng = np.random.default_rng(rng)
    nodes = np.array(list(G_ppi.nodes()))
    degs = np.array([G_ppi.degree(v) for v in nodes])

    tgt = [g for g in target_genes if g in G_ppi]
    if not tgt:
        return list(rng.choice(nodes, size=n, replace=False))
    tgt_degs = np.array([G_ppi.degree(g) for g in tgt])

    # bin edges over global degree range
    edges = np.quantile(degs, np.linspace(0, 1, bins + 1))
    # assign targets to bins
    t_bins = np.digitize(tgt_degs, edges[1:-1], right=True)

    sampled = []
    for b in t_bins:
        in_bin = nodes[np.digitize(degs, edges[1:-1], right=True) == b]
        if len(in_bin) == 0:
            in_bin = nodes
        sampled.append(rng.choice(in_bin))
    sampled = list(dict.fromkeys(sampled))  # unique preserve order
    if len(sampled) < n:
        remaining = n - len(sampled)
        sampled += list(rng.choice(nodes, size=remaining, replace=False))
    return sampled[:n]

def score_pathways(
    G_ppi: nx.Graph,
    pathways: dict[str, list[str]],
    target_genes: list[str],
    n_bootstrap: int = 500,
    degree_bins: int = 20,
    rng: int | None = 0,
) -> list[PathwayScore]:
    rng = np.random.default_rng(rng)
    tgt = [g for g in target_genes if g in G_ppi]
    m = len(tgt)
    if m == 0:
        raise ValueError("No target genes found in PPI graph.")

    # reference distribution of dclosest for degree-matched random gene sets
    ref = []
    for _ in range(n_bootstrap):
        rand_targets = degree_matched_bootstrap_genes(G_ppi, tgt, n=m, bins=degree_bins, rng=rng)
        # use a random pathway-size gene set? paper says reference distribution without depending on pathway size;
        # we keep pathway fixed and randomize targets (degree/size matched).
        ref.append(rand_targets)

    # compute z-scores
    scores = []
    for pid, genes in pathways.items():
        d = dclosest(G_ppi, genes, tgt)
        # compute bootstrap distances
        ds = [dclosest(G_ppi, genes, rt) for rt in ref]
        mu, sd = float(np.mean(ds)), float(np.std(ds) + 1e-8)
        z = (d - mu) / sd
        scores.append(PathwayScore(pid, d, z))

    # Sort by z ascending (more negative => closer than expected) OR descending depends on sign convention.
    # In Guney et al. (2016a)-style proximity, smaller distance is better, so we sort by ascending z.
    scores.sort(key=lambda s: s.z)
    return scores
