from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from drgat.data.datasets import load_expression_labels, read_data
from drgat.data.pathway_selector import PathwaySelector
#from drgat.eval.metrics import set_seed
from drgat.train.train_table1 import train_and_eval_drgat
from drgat.utils.ppi import load_ppi_graph


def _resolve_template(p: str, drug: str) -> str:
    return p.format(drug=drug)


def load_split(cfg: dict, split: str, drug: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load expression + labels for a given split.

    This mirrors the assumptions used by `generate_table1.py`.

    Expression CSV:
      - rows: sample_id (index or a column named sample_id)
      - columns: gene symbols

    Labels CSV:
      - columns: sample_id, response (0/1). 'drug' column is optional.
    """
    scfg = cfg["data"]["splits"][split]
    expr_path = Path(_resolve_template(scfg["expression"], drug))
    lab_path = Path(_resolve_template(scfg["labels"], drug))

    expr = pd.read_csv(expr_path)
    if "sample_id" in expr.columns:
        expr = expr.set_index("sample_id")
    else:
        expr = expr.set_index(expr.columns[0])

    labels = pd.read_csv(lab_path)
    if "drug" in labels.columns:
        labels = labels[labels["drug"].astype(str).str.lower() == drug.lower()].copy()
    if "response" not in labels.columns:
        raise ValueError(f"{lab_path} must contain a 'response' column (0/1).")
    if "sample_id" not in labels.columns:
        raise ValueError(f"{lab_path} must contain a 'sample_id' column.")
    labels = labels[["sample_id", "response"]].copy()

    common = expr.index.intersection(labels["sample_id"])
    expr = expr.loc[common].copy()
    labels = labels[labels["sample_id"].isin(common)].copy()
    labels = labels.set_index("sample_id").loc[common].reset_index()
    return expr, labels


def load_targets(cfg: dict, drug: str) -> list[str]:
    """Optional: load a per-drug list of target genes for feature selection."""
    tcfg = cfg["data"].get("targets")
    if tcfg is None:
        return []
    if isinstance(tcfg, dict) and "template" in tcfg:
        p = Path(_resolve_template(tcfg["template"], drug))
        txt = p.read_text().strip().splitlines()
        genes: list[str] = []
        for line in txt:
            line = line.strip()
            if not line:
                continue
            genes += [g.strip() for g in line.split(",") if g.strip()]
        return genes
    if isinstance(tcfg, dict) and drug in tcfg:
        v = tcfg[drug]
        return v if isinstance(v, list) else [str(v)]
    return []


def run(cfg: Dict[str, Any], out_dir: Path, drug_filter: str) -> Dict[str, Any]:
    """Generate DRGAT rows for Table 3 (CCLE/CTRP PR-AUCs).

    In the paper, Table 3 reports PR-AUCs over a large panel of drugs on
    CCLE/CTRP validation.

    This runner computes the **DRGAT** and **DRGAT (No-Aug)** columns only.
    Baselines (Super.FELT, DeepInsight, GraphCDR, ...) are not implemented.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    drug = drug_filter or cfg.get("drug")
    if not drug:
        raise ValueError("Table 5 generator requires --drug or cfg['drug'].")
    
    #set_seed(int(cfg.get("seed", 42)))
    device = cfg.get("device", "auto")

    expr_df, labels_df, ppi_df, pathways_df, target_genes = read_data(drug)
    
    #ppi_df = pd.read_csv(Path(cfg["data"]["ppi"]))
    #pathways_df = pd.read_csv(Path(cfg["data"]["pathways"]))

    G = load_ppi_graph(
        ppi_df,
        confidence_threshold=float(cfg["params"].get("confidence_threshold", 0.7)),
        take_lcc=True,
    )

    drugs = cfg["data"]["drugs"]
    table_rows = []

    train_split = cfg["data"].get("train_split", "ccle")
    test_split = cfg["data"].get("test_split", "ctrp")

    for drug in drugs:
        #train_expr, train_labels = load_split(cfg, train_split, drug)
        #test_expr, test_labels = load_split(cfg, test_split, drug)

        #target_genes = load_targets(cfg, drug)

        train_expr=expr_df.sample(frac=0.8)
        train_labels = labels_df[labels_df["sample_id"].isin(train_expr.index)].copy()
        test_expr=expr_df.sample(frac=0.2)
        test_labels = labels_df[labels_df["sample_id"].isin(test_expr.index)].copy()
        
        selector = PathwaySelector(G, pathways_df)
        selected_pathways, gene_set, distances = selector.select_for_drug(
            drug=drug,
            target_genes=target_genes,
            k_ratio=float(cfg["params"].get("k_pathways_ratio", 0.05)),
            n_boot=int(cfg["params"].get("n_boot", 50)),
        )

        # DRGAT (Aug)
        res_aug = train_and_eval_drgat(
            train_expr=train_expr,
            train_labels=train_labels,
            test_expr=test_expr,
            test_labels=test_labels,
            drug=drug,
            G=G,
            pathways_df=pathways_df,
            selected_pathways=selected_pathways,
            gene_set=gene_set,
            distances=distances,
            augment_ratio=float(cfg["params"].get("augment_ratio", 0.7)),
            #seed=int(cfg.get("seed", 42)),
            device=device,
            out_dir=out_dir / drug / f"drgat_aug_{train_split}_to_{test_split}",
        )

        # DRGAT (No-Aug)
        res_noaug = train_and_eval_drgat(
            train_expr=train_expr,
            train_labels=train_labels,
            test_expr=test_expr,
            test_labels=test_labels,
            drug=drug,
            G=G,
            pathways_df=pathways_df,
            selected_pathways=selected_pathways,
            gene_set=gene_set,
            distances=distances,
            augment_ratio=0.0,
            #seed=int(cfg.get("seed", 42)),
            device=device,
            out_dir=out_dir / drug / f"drgat_noaug_{train_split}_to_{test_split}",
        )

        table_rows.append(
            {
                "drug": drug,
                "pr_auc_drgat": res_aug.get("test_pr_auc"),
                "pr_auc_drgat_noaug": res_noaug.get("test_pr_auc"),
                "roc_auc_drgat": res_aug.get("test_roc_auc"),
                "roc_auc_drgat_noaug": res_noaug.get("test_roc_auc"),
                "n_train": res_aug.get("n_train"),
                "n_test": res_aug.get("n_test"),
            }
        )

    df = pd.DataFrame(table_rows)
    df.to_csv(out_dir / "table3_drgat_only.csv", index=False)

    summary = {
        "train_split": train_split,
        "test_split": test_split,
        "rows": table_rows,
        "params": cfg.get("params", {}),
        "seed": cfg.get("seed", 42),
    }
    (out_dir / "table3_drgat_only.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate DRGAT rows for Table 3 (CCLE/CTRP PR-AUCs).")
    ap.add_argument("--config", type=str, required=True, help="YAML config with split paths.")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--drug", type=str, default=None, help="If set, generate only for this drug.")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = run(cfg, Path(args.output_dir), drug_filter=args.drug)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
