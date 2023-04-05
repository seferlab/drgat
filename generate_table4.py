from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import math

from drgat.data.datasets import load_expression_labels, read_data
from drgat.data.pathway_selector import PathwaySelector
#from drgat.eval.metrics import set_seed
from drgat.train.train_table1 import train_and_eval_drgat
from drgat.utils.ppi import load_ppi_graph


def _resolve_template(p: str, drug: str) -> str:
    return p.format(drug=drug)


def load_split(cfg: dict, split: str, drug: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    tcfg = cfg["data"].get("targets")
    if tcfg is None:
        return []
    if isinstance(tcfg, dict) and "template" in tcfg:
        p = Path(_resolve_template(tcfg["template"], drug))
        if not p.exists():
            return []
        df = pd.read_csv(p)
        if "gene" in df.columns:
            return [str(x).strip() for x in df["gene"].dropna().tolist()]
        return [str(x).strip() for x in df.iloc[:, 0].dropna().tolist()]
    if isinstance(tcfg, dict) and "map" in tcfg:
        return [str(x).strip() for x in tcfg["map"].get(drug, [])]
    return []


def run(cfg: dict, out_dir: Path, drug_filter: str | None = None) -> Dict[str, Any]:
    #seed = int(cfg.get("seed", 42))
    #set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    drug = drug_filter or cfg.get("drug")
    if not drug:
        raise ValueError("Table 4 generator requires --drug or cfg['drug'].")

    #train_expr, train_labels = load_split(cfg, cfg["data"]["train_split"], drug)
    #pdx_expr, pdx_labels = load_split(cfg, cfg["data"]["test_splits"]["pdx"], drug)
    #tcga_expr, tcga_labels = load_split(cfg, cfg["data"]["test_splits"]["tcga"], drug)

    #ppi_path = Path(_resolve_template(cfg["data"]["ppi"], drug))
    #pathways_path = Path(_resolve_template(cfg["data"]["pathways"], drug))
    #ppi_df = pd.read_csv(ppi_path)
    #pathways_df = pd.read_csv(pathways_path)
    #target_genes = load_targets(cfg, drug)

    expr_df, labels_df, ppi_df, pathways_df, target_genes = read_data(drug)

    train_expr=expr_df.sample(frac=0.8)
    train_labels = labels_df[labels_df["sample_id"].isin(train_expr.index)].copy()
    pdx_expr=expr_df.sample(frac=0.2)
    pdx_labels = labels_df[labels_df["sample_id"].isin(pdx_expr.index)].copy()
    tcga_expr=expr_df.sample(frac=0.2)
    tcga_labels = labels_df[labels_df["sample_id"].isin(tcga_expr.index)].copy()
    
    #train_expr=expr_df
    #train_labels=labels_df
    #pdx_expr=expr_df
    #pdx_labels=labels_df
    #tcga_expr=expr_df
    #tcga_labels=labels_df
    
    G = load_ppi_graph(
        ppi_df,
        confidence_threshold=float(cfg.get("params", {}).get("confidence_threshold", 0.7)),
        take_lcc=True,
    )

    selector = PathwaySelector(G, pathways_df)
    selected_pathways, gene_set, distances = selector.select_for_drug(
        drug=drug,
        target_genes=target_genes,
        k_ratio=float(cfg.get("params", {}).get("k_pathways_ratio", 0.05)),
        n_boot=int(cfg.get("params", {}).get("n_boot", 5)),
    )

    base_aug = float(cfg.get("params", {}).get("augment_ratio", 0.7))
    variants = [
        ("Latent_Diff_on_AE", dict(augment_ratio=base_aug, aug_method="latent_ae", predictor_type="hogat", use_fs=True)),
        ("Diff_Direct", dict(augment_ratio=base_aug, aug_method="diff_direct", predictor_type="hogat", use_fs=True)),
        ("Latent_Diff_on_VAE", dict(augment_ratio=base_aug, aug_method="latent_vae", predictor_type="hogat", use_fs=True)),
        ("DRGAT_NoFS", dict(augment_ratio=base_aug, aug_method="latent_ae", predictor_type="hogat", use_fs=False)),
        ("DRGAT_NoHO_GAT", dict(augment_ratio=base_aug, aug_method="latent_ae", predictor_type="simple_gat", use_fs=True)),
        ("DRGAT_NoAug", dict(augment_ratio=0.0, aug_method="latent_ae", predictor_type="hogat", use_fs=True)),
        ("DRGAT", dict(augment_ratio=base_aug, aug_method="latent_ae", predictor_type="hogat", use_fs=True)),
    ]

    device = cfg.get("device", "auto")

    print(drug)
    rows = []
    for name, v in variants:
        print(name,v)
        if v.get("use_fs", True):
            sp, gs, dist = selected_pathways, gene_set, distances
        else:
            sp = sorted(pathways_df["pathway_id"].astype(str).unique().tolist())
            gs = [g for g in train_expr.columns if g in G]
            dist = {pid: 1.0 for pid in sp}

        subdir = out_dir / drug / name.lower()
        subdir.mkdir(parents=True, exist_ok=True)

        res_pdx = train_and_eval_drgat(
            train_expr=train_expr,
            train_labels=train_labels,
            test_expr=pdx_expr,
            test_labels=pdx_labels,
            drug=drug,
            G=G,
            pathways_df=pathways_df,
            selected_pathways=sp,
            gene_set=gs,
            distances=dist,
            augment_ratio=float(v["augment_ratio"]),
            #aug_method=str(v["aug_method"]),
            #predictor_type=str(v["predictor_type"]),
            #seed=seed,
            device=device,
            out_dir=subdir / "pdx",
        )
        res_tcga = train_and_eval_drgat(
            train_expr=train_expr,
            train_labels=train_labels,
            test_expr=tcga_expr,
            test_labels=tcga_labels,
            drug=drug,
            G=G,
            pathways_df=pathways_df,
            selected_pathways=sp,
            gene_set=gs,
            distances=dist,
            augment_ratio=float(v["augment_ratio"]),
            #aug_method=str(v["aug_method"]),
            #predictor_type=str(v["predictor_type"]),
            #seed=seed,
            device=device,
            out_dir=subdir / "tcga",
        )

        f1_pdx = res_pdx.get("test_f1")
        f1_tcga = res_tcga.get("test_f1")
        f1_mean = None
        if f1_pdx is not None and f1_tcga is not None:
            f1_mean = 0.5 * (float(f1_pdx) + float(f1_tcga))

        rows.append({"drug": drug, "variant": name, "pdx_f1": round(f1_pdx,2), "tcga_f1": round(f1_tcga,2), "mean_f1": round(f1_mean,2)})
    print(rows)
        
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table4_ablation_f1.csv", index=False)
    summary = {"drug": drug, "rows": rows, "params": cfg.get("params", {})}
    (out_dir / "table4_ablation_f1.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate DRGAT ablation rows for Table 4 (PDX/TCGA F1).")
    ap.add_argument("--config", type=str, required=True, help="YAML config with split paths.")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--drug", type=str, default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = run(cfg, Path(args.output_dir), drug_filter=args.drug)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
