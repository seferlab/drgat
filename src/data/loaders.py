from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

@dataclass
class DrugDataset:
    X: np.ndarray          # samples x genes
    y: np.ndarray          # 0/1 resistant/sensitive (or vice versa)
    genes: list[str]       # gene symbols aligned to X columns
    sample_ids: list[str]

def load_moli_gene_expression(raw_dir: Path, split: str, drug: str) -> DrugDataset:
    """Load gene expression and labels.

    The exact file names in the Zenodo record may differ by version.
    This loader is intentionally conservative: it searches within extracted
    archives in `raw_dir` for the requested split+drug.

    Expected user action:
      - Extract downloaded tar.gz archives into data/raw/extracted/...
      - Provide a mapping in configs if file naming differs.

    TODO: adapt to the exact MOLI Zenodo file structure once confirmed.
    """
    raw_dir = Path(raw_dir)
    raise NotImplementedError(
        "Please implement mapping to your extracted MOLI files. "
        "See src/data/README_data_mapping.md for guidance."
    )
