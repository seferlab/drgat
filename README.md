# DRGAT: Diffusion-based Graph Attention for Drug Response Prediction

This repository is a **reference implementation** of the pipeline described in the DRGAT paper (diffusion-based augmentation + HO-GAT classifier over pathway subgraphs). It is written to run on **real** gene expression and drug response datasets if you provide the paths, but it also includes a small **toy example** you can run end-to-end.

> ⚠️ Notes
> - This implementation mirrors the paper's modules: pathway selection (proximity + z-score), graph autoencoder, DDIM-style latent generator (MLP backbone), and HO-GAT (multi-hop) predictor.
> - External datasets (GDSC/CCLE/CTRP/TCGA/PDX, STRING/KEGG) are **not** included. Data loaders expect CSV files with documented schemas below.
> - The code uses **PyTorch** and **PyTorch Geometric**.

## Quick Start (toy run)

```bash
pip install -r requirements.txt
python run_pipeline.py --mode toy
```

This will:
1) Create a synthetic pathway graph and toy gene expression labels,
2) Train a graph autoencoder to compress gene expression,
3) Train a DDIM latent generator (conditional on resistant/sensitive),
4) Generate augmented samples,
5) Train an HO-GAT classifier on pathway subgraphs,
6) Print evaluation metrics.

## Run on real data

Prepare these inputs (CSV/TSV):
- `expression.csv`: rows=samples, columns=genes; **index column** named `sample_id`.
- `labels.csv`: columns: `sample_id,drug,response` where `response` ∈ {0,1} (0: resistant, 1: sensitive).
- `ppi_edges.csv`: columns: `gene_u,gene_v,score` (STRING-style confidence; filter `score>=0.7` before use or let the loader filter).
- `pathways.csv`: columns: `pathway_id,gene` (KEGG-derived lists or any curated pathways).

Optional:
- Patient test sets (PDX/TCGA) in the same `expression.csv` / `labels.csv` format (can be separate files; see CLI).

### Example command

```bash
python run_pipeline.py   --mode real   --train-expression /path/to/GDSC/expression.csv   --train-labels /path/to/GDSC/labels.csv   --test-expression /path/to/PDX_TCGA/expression.csv   --test-labels /path/to/PDX_TCGA/labels.csv   --ppi /path/to/STRING/ppi_edges.csv   --pathways /path/to/pathways.csv   --drug Cetuximab   --k-pathways-ratio 0.05   --augment-ratio 0.7   --output-dir ./outputs/cetuximab
```

## File layout

- `run_pipeline.py`: CLI that orchestrates the full pipeline.
- `drgat/data/datasets.py`: CSV loaders for expression/labels and basic splits.
- `drgat/data/pathway_selector.py`: Pathway proximity with z-scores and bootstrap null; selects top-K pathways per drug.
- `drgat/models/graph_autoencoder.py`: GAE that encodes expression (node features) into latent vectors and reconstructs expression.
- `drgat/models/ddim_latent.py`: Conditional DDIM with an MLP backbone tailored for latent (tabular) vectors.
- `drgat/models/ho_gat.py`: Multi-layer GAT with Jumping Knowledge to emulate high-order neighbor propagation.
- `drgat/train/train_drgat.py`: End-to-end training coordinating GAE → DDIM → augmentation → HO-GAT.
- `drgat/train/train_ablation.py`: Hooks for ablations (no FS, no HO-GAT, direct diffusion, VAE latent, etc.).
- `drgat/eval/metrics.py`: ROC-AUC, PR-AUC, F1, accuracy; synthetic-vs-real metrics (KLD, cosine, pairwise distance, log-cluster).
- `drgat/eval/evaluate_synthetic.py`: Quant/qual metrics for generated vs real distributions.
- `drgat/utils/ppi.py`: Build graphs from PPI edges, filter confidence, largest connected component.
- `drgat/utils/graph.py`: Helpers to build pathway subgraphs, distance-to-target computation, and batching utilities.

## Data schemas

**expression.csv**
```
sample_id,geneA,geneB,...
S1,0.12,1.03,...
S2,-0.04,0.88,...
```

**labels.csv**
```
sample_id,drug,response
S1,Docetaxel,1
S2,Docetaxel,0
```

**ppi_edges.csv**
```
gene_u,gene_v,score
TP53,MDM2,0.92
EGFR,GRB2,0.88
```

**pathways.csv**
```
pathway_id,gene
MAPK,EGFR
MAPK,GRB2
```

## Reproducibility & Hyperparameters

- Default `--augment-ratio 0.7` and `--k-pathways-ratio 0.05` reflect the paper’s best ranges.
- 5-fold stratified CV for early stopping & selection; patience=10 by default.
- Random seed fixed unless overridden.

## License

MIT (for this reference code). Please check original dataset licenses before use.
