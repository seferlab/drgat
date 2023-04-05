from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    # Guard: AUC undefined if one class present
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    return out
