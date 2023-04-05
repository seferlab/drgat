
import pandas as pd
import numpy as np
import os

def load_expression_labels(expr_path, labels_path, drug):
    expr = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(labels_path)
    labels = labels[labels["drug"] == drug].copy()
    labels = labels.set_index("sample_id")
    # Align
    common = expr.index.intersection(labels.index)
    expr = expr.loc[common]
    labels = labels.loc[common]
    return expr, labels.reset_index()

def read_data(drug, default="Docetaxel"):
    fname = "data/expression_{0}.csv".format(drug)

    expr_df = pd.read_csv("data/expression_{0}.csv".format(drug), index_col="sample_id")
    labels_df = pd.read_csv("data/labels_{0}.csv".format(drug))
    ppi_df = pd.read_csv("data/ppi_{0}.csv".format(drug))
    pathways_df = pd.read_csv("data/pathways_{0}.csv".format(drug))

    target_genes = []
    with open("data/target_{0}.csv".format(drug), "r") as infile:
        for line in infile:
            line = line.rstrip()
            target_genes.append(line)
                
    return expr_df, labels_df, ppi_df, pathways_df, target_genes

