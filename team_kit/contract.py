"""
team_kit.contract  --  THE SHARED AGREEMENT (frozen at kickoff)
================================================================

This is the ONLY thing the three groups must agree on. After this is frozen,
each group works entirely on its own file (see group_*_starter.py) using the
synthetic data + mocks, and nobody needs to talk to anyone until integration.

It defines:
  * the data schemas + the flow_id join key,
  * the shared DATA ACCESS functions (everyone samples the SAME way),
  * the three model INTERFACES (Protocols) each group implements,
  * the ARTIFACT format each model group ships (preprocessor + model bundled),
  * the METRICS format the evaluation/report consumes.

NOTE: preprocessing is NOT shared -- each model group writes its own and ships
the *fitted* transformer bundled with its model (see save_bundle). The cascade
group never re-fits; it loads the bundles and calls .transform()/.predict().
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd

# Make the repo root importable so we can reuse the shared data layer.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config                                   # noqa: E402
from src import data_loader, flow_aggregation, sampling  # noqa: E402

# --------------------------------------------------------------------------- #
# 1. SCHEMA  (the column contract -- identical for real and synthetic data)
# --------------------------------------------------------------------------- #
ATTACK_TYPES = config.ATTACK_TYPES               # the five classes
BENIGN_LABEL = config.BENIGN_LABEL
ALL_CLASSES = config.ALL_CLASSES

# Label columns present in every sample (added by the data layer):
#   Label       : int  0=benign, 1=attack
#   attack_type : str  one of ALL_CLASSES
LABEL_COL = "Label"
ATTACK_TYPE_COL = "attack_type"

# The join key between a packet and its flow. CANONICAL + direction-independent:
#   sorted("{srcIP}:{srcPort}", "{dstIP}:{dstPort}") joined by "-".
# It is built by the data layer; never rebuild it differently in a group.
FLOW_ID_COL = "flow_id"

# --------------------------------------------------------------------------- #
# 2. DATA ACCESS  (everyone gets identical inputs from these three functions)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _populations(seed: int):
    """Load (and cache) the packet + flow populations once per seed."""
    packets, flows, source = data_loader.load_populations(seed=seed, verbose=False)
    return packets, flows, source


def get_packet_sample(seed: int = 42) -> pd.DataFrame:
    """Phase-2 input: a sampled packet dataset (Task 1.1 proportions)."""
    packets, _, _ = _populations(seed)
    return sampling.sample_dataset(packets, seed=seed, verbose=False)


def get_flow_sample(seed: int = 42) -> pd.DataFrame:
    """Phase-3 input: one record per flow (segments aggregated), then sampled."""
    _, flows, _ = _populations(seed)
    unified = flow_aggregation.aggregate_flows(flows, verbose=False)
    return sampling.sample_dataset(unified, seed=seed, verbose=False)


def get_packet_and_flows(seed: int = 42):
    """Cascade input: (sampled packet df, full unified-flow table to join on)."""
    packets, flows, _ = _populations(seed)
    unified = flow_aggregation.aggregate_flows(flows, verbose=False)
    pkt = sampling.sample_dataset(packets, seed=seed, verbose=False)
    return pkt, unified


# --------------------------------------------------------------------------- #
# 3. MODEL INTERFACES  (each group implements ONE of these)
# --------------------------------------------------------------------------- #
@runtime_checkable
class FeatureTransformer(Protocol):
    """Each group's own preprocessing. Fit once on train, apply everywhere."""
    feature_names: list
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray: ...
    def transform(self, df: pd.DataFrame) -> np.ndarray: ...


@runtime_checkable
class AnomalyDetector(Protocol):
    """GROUP 'UNSUPERVISED' implements this. NOTE: fit() takes NO labels."""
    def fit(self, X: np.ndarray) -> "AnomalyDetector": ...
    def anomaly_score(self, X: np.ndarray) -> np.ndarray: ...   # higher = worse
    def predict(self, X: np.ndarray) -> np.ndarray: ...         # 1 = alert/anomaly


@runtime_checkable
class SignatureClassifier(Protocol):
    """GROUP 'SUPERVISED' implements this. fit() uses labels; test() does not."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SignatureClassifier": ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...   # P(attack) in [0,1]
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# 4. ARTIFACT FORMAT  (how a model group hands its work to the cascade group)
# --------------------------------------------------------------------------- #
@dataclass
class Bundle:
    """A fitted (preprocessor + model) pair. THE unit of handoff between groups."""
    kind: str                 # "unsupervised" or "supervised"
    preprocessor: object      # a FeatureTransformer (already fitted)
    model: object             # an AnomalyDetector or SignatureClassifier (fitted)
    extra: dict               # e.g. {"threshold": 0.5}; free-form


def save_bundle(path: str, bundle: Bundle) -> None:
    joblib.dump(bundle, path)
    print(f"[bundle] saved {bundle.kind} -> {path}")


def load_bundle(path: str) -> Bundle:
    return joblib.load(path)


# Agreed file names the cascade group looks for at integration time.
UNSUP_BUNDLE = os.path.join(_ROOT, "team_kit", "artifacts", "unsupervised.joblib")
SUP_BUNDLE = os.path.join(_ROOT, "team_kit", "artifacts", "supervised.joblib")


# --------------------------------------------------------------------------- #
# 5. METRICS FORMAT  (binary_metrics returns exactly these keys)
# --------------------------------------------------------------------------- #
# {precision, recall, f1, accuracy, fpr, fnr, tp, fp, tn, fn, [auc_roc]}
# Use src.evaluate.binary_metrics / per_attack_detection_rate so every group's
# numbers are directly comparable. (Imported lazily to keep this module light.)
def binary_metrics(y_true, y_pred, scores=None) -> dict:
    from src import evaluate
    return evaluate.binary_metrics(y_true, y_pred, scores)


def per_attack_detection_rate(y_multi, y_pred) -> dict:
    from src import evaluate
    return evaluate.per_attack_detection_rate(y_multi, y_pred)
