
import pandas as pd
import numpy as np

def load_expression_labels(expr_path, labels_path, drug):
    expr = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(labels_path)
    labels = labels[labels["drug"] == drug].copy()
    labels = labels.set_index("sample_id")
    # Align
    common = expr.index.intersection(labels.index)
    expr = expr.loc[common]
    labels = labels.loc[common]
    return expr, labels.reset_index()

def make_toy_data(n_samples=120, n_genes=400, n_pathways=50, seed=42):
    rng = np.random.default_rng(seed)
    # random gene names
    genes = [f"G{g}" for g in range(n_genes)]
    # expression
    X = rng.normal(0, 1, size=(n_samples, n_genes))
    # latent factors affect subset of genes
    grp = rng.choice([0,1], size=n_samples, p=[0.5,0.5])
    for i in range(n_samples):
        if grp[i] == 1:
            X[i, :50] += rng.normal(0.8, 0.5, size=50)
        else:
            X[i, :50] += rng.normal(-0.8, 0.5, size=50)
    expr_df = pd.DataFrame(X, columns=genes)
    expr_df.insert(0, "sample_id", [f"S{i}" for i in range(n_samples)])
    expr_df = expr_df.set_index("sample_id")
    # labels
    labels_df = pd.DataFrame({
        "sample_id": expr_df.index,
        "drug": ["Docetaxel"]*n_samples,
        "response": grp.astype(int),
    })
    # PPI edges (chain + random)
    rows = []
    for i in range(n_genes-1):
        rows.append(("G"+str(i), "G"+str(i+1), 0.9))
    for _ in range(2*n_genes):
        u, v = rng.choice(genes, size=2, replace=False)
        rows.append((u,v, float(rng.uniform(0.7, 0.95))))
    ppi_df = pd.DataFrame(rows, columns=["gene_u","gene_v","score"])
    # Pathways: random partitions
    pw_rows = []
    for p in range(n_pathways):
        k = rng.integers(10, 25)
        members = rng.choice(genes, size=k, replace=False)
        for g in members:
            pw_rows.append((f"PW{p}", g))
    pathways_df = pd.DataFrame(pw_rows, columns=["pathway_id","gene"])
    target_genes = ["G1","G2","G3"]
    return expr_df, labels_df, ppi_df, pathways_df, target_genes
