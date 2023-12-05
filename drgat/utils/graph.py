
import torch, networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

def build_pathway_subgraphs(G, pathways, selected_ids, gene_to_idx, node_features):
    subgraphs = []
    for pid in selected_ids:
        genes = [g for g in pathways[pid] if g in G and g in gene_to_idx]
        H = G.subgraph(genes).copy()
        if H.number_of_nodes() == 0: 
            continue
        # attach node features (in order of from_networkx mapping)
        for n in H.nodes:
            idx = gene_to_idx[n]
            H.nodes[n]["x"] = torch.tensor(node_features[idx], dtype=torch.float)
        data = from_networkx(H)
        subgraphs.append((pid, data))
    return subgraphs

def make_gene_lookup(genes):
    return {g:i for i,g in enumerate(genes)}

def inv_distance_weight(d):
    return 1.0/float(d) if d>0 else 1.0
