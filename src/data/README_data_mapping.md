# Data mapping notes (MOLI Zenodo)

The DRGAT paper uses the same dataset configuration as MOLI:
- Train: GDSC cell lines (gene expression + drug response labels)
- Test: PDX + TCGA (gene expression + response labels)
- Labels: experimentally determined cutoffs from Iorio et al. (2016)

The Zenodo record (4036592) contains multiple tar.gz archives (omics types, splits, etc).
Because Zenodo versions and file naming can vary, this repo does not hardcode paths.

## What you need to do once (5–10 minutes)

1) Download & extract into:
```
data/raw/
  <zenodo files...>
data/raw/extracted/
  <folders with expression matrices and response labels>
```

2) Open `src/data/loaders.py` and implement `load_moli_gene_expression(...)` by pointing to:
- expression matrix for the requested `split` (gdsc_train, pdx_test, tcga_test)
- response labels for the requested `drug`

3) Run:
```
python run_pipeline.py train_eval --config configs/table1_gdsc_pdx_tcga.yaml --output-dir outputs/table1
```

If you paste (or upload) the Zenodo file list / folder tree, I can fill in the loader automatically.
