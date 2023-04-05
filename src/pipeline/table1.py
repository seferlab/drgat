from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from ..data.loaders import load_moli_gene_expression
from ..eval.metrics import compute_metrics

def run_table1_experiment(cfg: dict, out_dir: Path) -> None:
    """Runs the Table-1-like setup:
    - Train on GDSC
    - Validate on GDSC (80:20, 5-fold CV for hyperparams; not fully implemented here)
    - Test on PDX and TCGA
    """
    drugs = cfg["data"]["drugs"]

    results = {}

    for drug in drugs:
        # TODO: implement real training
        # Placeholder: once data loaders are wired, train DRGAT and output predicted probs.
        # Here we only demonstrate the expected outputs format.
        results[drug] = {
            "pdx": {"auc": None},
            "tcga": {"auc": None},
        }

    (out_dir / "table1_results.json").write_text(json.dumps(results, indent=2))
    print(f"[OK] Wrote {out_dir/'table1_results.json'}")

    print("\nNEXT STEPS:")
    print("1) Implement src/data/loaders.py mapping to your extracted MOLI Zenodo files.")
    print("2) Implement model training in src/pipeline/table1.py (DRGAT AE + DDIM + HO-GAT).")
    print("If you share the extracted file tree, I can finish those parts quickly.")
