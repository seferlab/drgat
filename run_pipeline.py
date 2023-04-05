#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import generate_table1
import generate_table2
import generate_table3
import generate_table4
import generate_table5


def parse_args():
    p = argparse.ArgumentParser(
        description="Run DRGAT pipeline to generate Tables 1–4 (DRGAT-side results) for a single drug."
    )
    p.add_argument("--drug", type=str, default="Docetaxel", help="Drug name, e.g., Docetaxel")
    p.add_argument("--output-dir", type=str, default="outputs/all_tables", help="Root output directory")
    p.add_argument("--table1-config", type=str, default="configs/table1_paths_example.yaml")
    p.add_argument("--table2-config", type=str, default="configs/table2_paths_example.yaml")
    p.add_argument("--table3-config", type=str, default="configs/table3_paths_example.yaml")
    p.add_argument("--table4-config", type=str, default="configs/table4_paths_example.yaml")
    p.add_argument("--table5-config", type=str, default="configs/table5_paths_example.yaml")
    p.add_argument("--skip", nargs="*", default=[], choices=["1", "2", "3", "4", "5"], help="Skip specific tables")
    p.add_argument("--style", type=str, default="1")
    return p.parse_args()


def _load_yaml(p: str) -> dict:
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {p}. Please edit the example configs under ./configs/ to point to your data."
        )
    return yaml.safe_load(path.read_text())


def main():
    args = parse_args()
    out_root = Path(args.output_dir) / args.drug
    out_root.mkdir(parents=True, exist_ok=True)

    results = {"drug": args.drug, "outputs": {}}

    if args.style == "1":
        cfg1 = _load_yaml(args.table1_config)
        out1 = out_root / "table1"
        out = generate_table1.run(cfg1, out1, drug_filter=args.drug)
        results["outputs"]["table1"] = {"dir": str(out1), "summary": out}

    if args.style == "2":
        cfg2 = _load_yaml(args.table2_config)
        out2 = out_root / "table2"
        out = generate_table2.run(cfg2, out2, drug_filter=args.drug)
        results["outputs"]["table2"] = {"dir": str(out2), "summary": out}

    if args.style == "3":
        cfg3 = _load_yaml(args.table3_config)
        out3 = out_root / "table3"
        out = generate_table3.run(cfg3, out3, drug_filter=args.drug)
        results["outputs"]["table3"] = {"dir": str(out3), "summary": out}

    if args.style == "4":
        cfg4 = _load_yaml(args.table4_config)
        out4 = out_root / "table4"
        out = generate_table4.run(cfg4, out4, drug_filter=args.drug)
        results["outputs"]["table4"] = {"dir": str(out4), "summary": out}

    if args.style == "5":
        cfg5 = _load_yaml(args.table5_config)
        out5 = out_root / "table5"
        out = generate_table5.run(cfg5, out5, drug_filter=args.drug)
        results["outputs"]["table5"] = {"dir": str(out5), "summary": out}

    (out_root / "all_tables_summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
