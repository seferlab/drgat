# DRGAT: Diffusion-based Graph Attention for Drug Response Prediction

It implements the end-to-end pipeline described in the paper:
1. **Pathway/feature selection** via proximity of drug targets to biological pathways (shortest-path distance, z-scored with degree-matched bootstraps)
2. **Graph autoencoder** to compress pathway-structured gene-expression into a latent space
3. **Conditional latent DDIM** to augment latent samples for **sensitive/resistant** classes, then decode to expression
4. **HO-GAT predictor** (high-order neighbor propagation attention) over pathway subgraphs + target-protein distance features

> **Data**: Train/test configuration follows Sharifi-Noghabi et al. (2019) MOLI and that the data were downloaded from Zenodo.  
> The dataset record is “MOLI: multi-omics late integration …” on Zenodo.

## Quickstart

### 1) Create environment
```bash
conda create -n drgat python=3.12 -y
conda activate drgat
pip install -r requirements.txt
```

### 2) Download data
```bash
python run_pipeline_data.py download --zenodo-record 4036592 --data-dir data/raw
```

### 3) Run the pipeline

### Example command

```bash
python run_train_and_eval_drgat.py \
  --drug Docetaxel \
  --train-expr /path/to/gdsc_expr.csv --train-labels /path/to/gdsc_labels.csv \
  --test-expr  /path/to/pdx_expr.csv  --test-labels  /path/to/pdx_labels.csv \
  --ppi /path/to/ppi.csv --pathways /path/to/pathways.csv \
  --augment-ratio 0.7 --aug-method latent_ae --predictor-type hogat \
  --out-dir outputs/custom_run

python run_pipeline.py --style 1 --table1-config configs/table1_paths_example.yaml
python run_pipeline.py --style 2 --table2-config configs/table2_paths_example.yaml
python run_pipeline.py --style 4 --table4-config configs/table4_paths_example.yaml
python run_pipeline.py --style 5 --table5-config configs/table5_paths_example.yaml
python generate_table6.py --output-dir outputs/table6
```

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

## Notes

The paper uses:
- **80:20** train/validation split with **stratified 5-fold CV** for hyperparameter selection and early stopping (patience 10).  
- **Augmentation rate**: synthetic samples are ~**70–75%** of real sample count (best-performing range); the Table 1 experiments used **70%**.  
- **Pathway selection**: optimal was **5%** of pathways.  
These are encoded as defaults in `configs/*.yaml`.

## Repository layout

- `run_pipeline.py` — entry point, with subcommands
- `src/data/` — dataset download + loaders + splits
- `src/graphs/` — PPI + KEGG pathway graphs, distance computation, bootstrap z-scores
- `src/models/` — GraphAE, ConditionalDDIM, HOGAT, Predictor
- `src/train/` — training loops, early stopping, CV
- `src/eval/` — AUC/PR-AUC + synthetic-quality metrics (KLD, PD, Log-Cluster, cosine)
- `configs/` — experiment configs mirroring the paper’s setups
- `scripts/` — helper scripts

## Disclaimer

This repo is designed as the paper’s described methodology, but exact numeric reproduction may still depend on:
- the exact preprocessed files used in the referenced Zenodo release,
- the specific STRING and KEGG versions,
- random seeds and hardware (esp. for diffusion sampling).



