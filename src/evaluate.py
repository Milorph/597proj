"""
Evaluation, metrics and visualisation.

Computes every metric the brief requires (precision/recall/F1, per-attack
detection rate, FPR/FNR, AUC-ROC, confusion matrices, FP-reduction, combined
system accuracy) and renders the figures (ROC, PR, confusion matrices,
per-attack bars, cluster structure). All numeric results are also written to
JSON so the report can cite exact values.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, accuracy_score)

import config


# --------------------------------------------------------------------------- #
# Core metric blocks
# --------------------------------------------------------------------------- #
def binary_metrics(y_true, y_pred, scores=None) -> dict:
    """Precision/recall/F1/accuracy + FPR/FNR (+AUC if scores given)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if scores is not None:
        try:
            out["auc_roc"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            out["auc_roc"] = float("nan")
    return out


def per_attack_detection_rate(y_multi, y_pred) -> dict:
    """Recall (detection rate) for each attack type."""
    rates = {}
    for atk in config.ATTACK_TYPES:
        mask = (y_multi == atk)
        n = int(mask.sum())
        det = int(y_pred[mask].sum()) if n else 0
        rates[atk] = {"detected": det, "total": n,
                      "detection_rate": (det / n) if n else float("nan")}
    return rates


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"]); ax.set_yticklabels(["Benign", "Attack"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); _save(fig, fname)


def plot_roc(curves: dict, fname, title="ROC curves"):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for name, (y_true, scores) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, scores):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); _save(fig, fname)


def plot_pr(curves: dict, fname, title="Precision-Recall curves"):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for name, (y_true, scores) in curves.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ax.plot(rec, prec, label=f"{name} (AP-AUC={auc(rec, prec):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); _save(fig, fname)


def plot_per_attack(rates_phase2: dict, rates_phase3: dict, fname):
    atks = config.ATTACK_TYPES
    p2 = [rates_phase2[a]["detection_rate"] for a in atks]
    p3 = [rates_phase3[a]["detection_rate"] for a in atks]
    x = np.arange(len(atks)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w / 2, p2, w, label="Phase 2 only (packet, unsup.)")
    ax.bar(x + w / 2, p3, w, label="Two-stage (combined)")
    ax.set_xticks(x); ax.set_xticklabels(atks, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Detection rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-attack detection rate by phase"); ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, fname)


def plot_score_distributions(scores, y_true, threshold, fname):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(scores[y_true == 0], bins=80, alpha=0.6, label="Benign", density=True)
    ax.hist(scores[y_true == 1], bins=80, alpha=0.6, label="Attack", density=True)
    ax.axvline(threshold, color="red", ls="--", label=f"threshold={threshold:.2f}")
    ax.set_xlabel("Anomaly score"); ax.set_ylabel("Density")
    ax.set_title("Phase-2 anomaly-score distribution"); ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, fname)


def plot_cluster_composition(cluster_ids, y_multi, fname):
    """Stacked bar of class composition per k-means cluster (DoS/DDoS view)."""
    df = pd.DataFrame({"cluster": cluster_ids, "cls": y_multi})
    comp = df.groupby(["cluster", "cls"]).size().unstack(fill_value=0)
    comp = comp.reindex(columns=config.ALL_CLASSES, fill_value=0)
    frac = comp.div(comp.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(frac))
    for cls in config.ALL_CLASSES:
        ax.bar(frac.index.astype(str), frac[cls], bottom=bottom, label=cls)
        bottom += frac[cls].to_numpy()
    ax.set_xlabel("K-means cluster"); ax.set_ylabel("Class fraction")
    ax.set_title("Cluster composition (packet level)")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout(); _save(fig, fname)


def _save(fig, fname):
    config.ensure_dirs()
    # Save as SVG (text-based, renders inline on GitHub, transfers cleanly via
    # the GitHub contents API). The requested extension is normalised to .svg.
    fname = os.path.splitext(fname)[0] + ".svg"
    path = os.path.join(config.FIG_DIR, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {path}")


# --------------------------------------------------------------------------- #
# JSON dump
# --------------------------------------------------------------------------- #
def save_metrics(obj: dict, fname: str):
    config.ensure_dirs()
    path = os.path.join(config.METRICS_DIR, fname)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    print(f"[metrics] wrote {path}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
