#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple, List

import pandas as pd

# Ensure local imports work when running from repo root
import sys as _sys
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from drgat.data.pathway_selector import PathwaySelector
from drgat.utils.ppi import load_ppi_graph
from drgat.train.train_table1 import train_and_eval_drgat


def _load_expr(expr_path: Path) -> pd.DataFrame:
    df = pd.read_csv(expr_path)
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    else:
        # common pattern: first column is sample id
        df = df.set_index(df.columns[0])
    # ensure string index
    df.index = df.index.astype(str)
    return df


def _load_labels(labels_path: Path, drug: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    if "sample_id" not in df.columns:
        raise ValueError(f"{labels_path} must contain a 'sample_id' column.")
    if "response" not in df.columns:
        raise ValueError(f"{labels_path} must contain a 'response' column (0/1).")

    if drug and ("drug" in df.columns):
        df = df[df["drug"].astype(str).str.lower() == drug.lower()].copy()

    df = df[["sample_id", "response"]].copy()
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def load_split(expr_path: Path, labels_path: Path, drug: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    expr = _load_expr(expr_path)
    labels = _load_labels(labels_path, drug=drug)

    common = expr.index.intersection(labels["sample_id"])
    if len(common) == 0:
        raise ValueError(
            f"No overlapping sample_ids between expression ({expr_path}) and labels ({labels_path}). "
            f"Check that both use the same sample_id strings."
        )

    expr = expr.loc[common].copy()
    labels = labels[labels["sample_id"].isin(common)].copy()
    labels = labels.set_index("sample_id").loc[common].reset_index()
    return expr, labels


def _parse_targets(targets: str | None, targets_file: str | None) -> List[str]:
    genes: List[str] = []
    if targets_file:
        p = Path(targets_file)
        txt = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in txt:
            line = line.strip()
            if not line:
                continue
            # allow comma-separated lines too
            genes.extend([g.strip() for g in line.split(",") if g.strip()])
    if targets:
        genes.extend([g.strip() for g in targets.split(",") if g.strip()])
    # unique preserve order
    seen = set()
    out = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run train_and_eval_drgat directly with CLI parameters (single train/test split)."
    )
    ap.add_argument("--drug", required=True, help="Drug name (used for filtering labels if they have a 'drug' column).")

    ap.add_argument("--train-expr", required=True, type=str, help="CSV: training expression (rows=samples, cols=genes).")
    ap.add_argument("--train-labels", required=True, type=str, help="CSV: training labels with sample_id,response.")
    ap.add_argument("--test-expr", required=True, type=str, help="CSV: test expression.")
    ap.add_argument("--test-labels", required=True, type=str, help="CSV: test labels with sample_id,response.")

    ap.add_argument("--ppi", required=True, type=str, help="PPI edge list CSV (as used by other table scripts).")
    ap.add_argument("--pathways", required=True, type=str, help="Pathway gene-membership CSV.")

    ap.add_argument("--targets", default=None, type=str, help="Comma-separated target genes for pathway selection.")
    ap.add_argument("--targets-file", default=None, type=str, help="Text file with target genes (one per line or CSV).")

    ap.add_argument("--augment-ratio", default=0.7, type=float, help="Augmentation ratio (e.g., 0.7).")
    ap.add_argument("--aug-method", default="latent_ae",
                    choices=["latent_ae", "diff_direct", "latent_vae", "cvae", "ctgan", "tvae", "copulagan"],
                    help="Augmentation backbone (paper default: latent_ae).")
    ap.add_argument("--predictor-type", default="hogat", choices=["hogat", "gat"],
                    help="Predictor network (default: hogat).")

    ap.add_argument("--k-pathways-ratio", default=0.05, type=float,
                    help="Fraction of pathways to keep in PathwaySelector (paper uses a small ratio).")
    ap.add_argument("--n-boot", default=50, type=int, help="Bootstrap runs for pathway selection.")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--device", default="auto", type=str)

    ap.add_argument("--out-dir", required=True, type=str, help="Output directory for artifacts + metrics JSON.")
    ap.add_argument("--save-json", default="metrics.json", type=str, help="Filename for metrics JSON under out-dir.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    train_expr, train_labels = load_split(Path(args.train_expr), Path(args.train_labels), drug=args.drug)
    test_expr, test_labels = load_split(Path(args.test_expr), Path(args.test_labels), drug=args.drug)

    # Load PPI + pathways
    G = load_ppi_graph(Path(args.ppi))
    pathways_df = pd.read_csv(Path(args.pathways))

    # Select pathways + gene set
    target_genes = _parse_targets(args.targets, args.targets_file)
    selector = PathwaySelector(G, pathways_df)
    selected_pathways, gene_set, distances = selector.select_for_drug(
        drug=args.drug,
        target_genes=target_genes,
        k_ratio=float(args.k_pathways_ratio),
        n_boot=int(args.n_boot),
    )

    # Run
    res = train_and_eval_drgat(
        train_expr=train_expr,
        train_labels=train_labels,
        test_expr=test_expr,
        test_labels=test_labels,
        drug=args.drug,
        G=G,
        pathways_df=pathways_df,
        selected_pathways=selected_pathways,
        gene_set=gene_set,
        distances=distances,
        augment_ratio=float(args.augment_ratio),
        seed=int(args.seed),
        device=str(args.device),
        aug_method=str(args.aug_method),
        predictor_type=str(args.predictor_type),
        out_dir=out_dir,
    )

    # Write metrics
    metrics_path = out_dir / args.save_json
    metrics_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # Also print a one-line summary
    summary_keys = ["roc_auc", "pr_auc", "f1", "acc"]
    summary = {k: res.get(k) for k in summary_keys if k in res}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
