#!/usr/bin/env python3
import argparse, os, json, random, numpy as np, torch
from pathlib import Path
from drgat.data.datasets import load_expression_labels, read_data
from drgat.data.pathway_selector import PathwaySelector
from drgat.utils.ppi import load_ppi_graph
from drgat.train.train_drgat import train_full_pipeline
from drgat.eval.metrics import set_seed

def parse_args():
    p = argparse.ArgumentParser(description="DRGAT end-to-end pipeline")
    p.add_argument("--mode", choices=["toy","real"], default="real")
    p.add_argument("--train-expression", type=str, default=None)
    p.add_argument("--train-labels", type=str, default=None)
    p.add_argument("--test-expression", type=str, default=None)
    p.add_argument("--test-labels", type=str, default=None)
    p.add_argument("--ppi", type=str, default=None)
    p.add_argument("--pathways", type=str, default=None)
    p.add_argument("--drug", type=str, default="Docetaxel")
    p.add_argument("--target-genes", type=str, default=None, help="Comma-separated drug target genes (optional, used in proximity)")
    p.add_argument("--confidence-threshold", type=float, default=0.7)
    p.add_argument("--k-pathways-ratio", type=float, default=0.05, help="Select top-K ratio of pathways per drug")
    p.add_argument("--augment-ratio", type=float, default=0.7, help="Synthetic samples as fraction of total real per-class")
    p.add_argument("--output-dir", type=str, default="./outputs/run")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "real":
        #expr_df, labels_df, ppi_df, pathways_df, target_genes = make_toy_data()
        expr_df, labels_df, ppi_df, pathways_df, target_genes = read_data(args.drug)
        drug = labels_df["drug"].unique()[0]
    else:
        assert args.train_expression and args.train_labels and args.ppi and args.pathways, "Missing required real-data paths."
        expr_df, labels_df = load_expression_labels(args.train_expression, args.train_labels, drug=args.drug)
        ppi_df = None
        import pandas as pd
        ppi_df = pd.read_csv(args.ppi)
        pathways_df = pd.read_csv(args.pathways)
        drug = args.drug
        target_genes = [g.strip() for g in (args.target_genes.split(",") if args.target_genes else [])]

    # Build PPI graph
    G = load_ppi_graph(ppi_df, confidence_threshold=args.confidence_threshold, take_lcc=True)
    
    # Pathway selection (z-score proximity to drug targets)
    selector = PathwaySelector(G, pathways_df)
    selected_pathways, gene_set, distances = selector.select_for_drug(
        drug=drug,
        target_genes=target_genes,
        k_ratio=args.k_pathways_ratio,
        n_boot=5
    )

    # Train pipeline
    results = train_full_pipeline(
        expression_df=expr_df,
        labels_df=labels_df,
        drug=drug,
        G=G,
        pathways_df=pathways_df,
        selected_pathways=selected_pathways,
        gene_set=gene_set,
        distances=distances,
        output_dir=args.output_dir,
        augment_ratio=args.augment_ratio,
        seed=args.seed,
    )

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
