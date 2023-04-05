#!/usr/bin/env python
"""DRGAT reproduction runner.

Usage:
  python run_pipeline.py download --zenodo-record 4036592
  python run_pipeline.py train_eval --config configs/table1_gdsc_pdx_tcga.yaml --output-dir outputs/table1
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from src.utils.repro import set_global_seed
from src.data.download import download_zenodo_record
from src.pipeline.table1 import run_table1_experiment

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_dl = sub.add_parser("download", help="Download datasets from Zenodo.")
    ap_dl.add_argument("--zenodo-record", type=int, default=4036592)
    ap_dl.add_argument("--data-dir", type=str, default="data/raw")
    ap_dl.add_argument("--force", action="store_true")

    ap_te = sub.add_parser("train_eval", help="Run train+eval experiment defined by a config.")
    ap_te.add_argument("--config", type=str, required=True)
    ap_te.add_argument("--output-dir", type=str, required=True)

    args = ap.parse_args()

    if args.cmd == "download":
        out = Path(args.data_dir)
        out.mkdir(parents=True, exist_ok=True)
        download_zenodo_record(record_id=args.zenodo_record, out_dir=out, force=args.force)
        return

    if args.cmd == "train_eval":
        cfg = yaml.safe_load(Path(args.config).read_text())
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        seed = int(cfg.get("seed", 1337))
        set_global_seed(seed)
        task = cfg.get("task", "table1")
        if task == "table1":
            run_table1_experiment(cfg, out)
        else:
            raise ValueError(f"Unknown task: {task}")

if __name__ == "__main__":
    main()
