"""
GROUP: UNSUPERVISED  (Phase 2 -- anomaly-based IDS)
===================================================
You own this file + your own preprocessing. You implement an AnomalyDetector.
You NEVER use labels in fit() (labels are only for evaluation).

Run it standalone, no other group needed:
    python -m team_kit.group_unsupervised_starter

Deliverable to the cascade group: team_kit/artifacts/unsupervised.joblib
(a fitted preprocessor + detector bundle). A working IsolationForest baseline is
provided -- replace/extend it (autoencoder, k-means, ensemble) to improve recall.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from team_kit import contract


# ---- YOUR preprocessing (you own this; ships bundled with your model) ------- #
class PacketPreprocessor:
    feature_names: list = []

    def __init__(self):
        self.imp = SimpleImputer(strategy="median")
        self.sc = StandardScaler()

    def _mat(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = {contract.LABEL_COL, contract.ATTACK_TYPE_COL, contract.FLOW_ID_COL,
                "src_ip", "dst_ip", "timestamp", "segment_index"}
        num = df.select_dtypes(include="number")
        cols = [c for c in num.columns if c not in drop]
        return num[cols].replace([np.inf, -np.inf], np.nan)

    def fit_transform(self, df):
        X = self._mat(df)
        self.feature_names = list(X.columns)
        return self.sc.fit_transform(self.imp.fit_transform(X))

    def transform(self, df):
        X = df.reindex(columns=self.feature_names).replace([np.inf, -np.inf], np.nan)
        return self.sc.transform(self.imp.transform(X))


# ---- YOUR model: implement contract.AnomalyDetector ------------------------- #
class MyDetector:
    """Baseline. TODO: add autoencoder reconstruction error / k-means, tune."""
    def __init__(self, contamination=0.2, seed=42):
        self.contamination = contamination
        self.iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
        self._thr = 0.0

    def fit(self, X):                       # <-- NO labels here (Phase-2 rule)
        self.iso.fit(X)
        s = self.anomaly_score(X)
        self._thr = float(np.quantile(s, 1 - self.contamination))   # wide net
        return self

    def anomaly_score(self, X):
        return -self.iso.score_samples(X)   # higher = more anomalous

    def predict(self, X):
        return (self.anomaly_score(X) >= self._thr).astype(int)


def main():
    df = contract.get_packet_sample(seed=42)
    tr, te = train_test_split(df, test_size=0.3, random_state=42,
                              stratify=df[contract.ATTACK_TYPE_COL])

    prep = PacketPreprocessor()
    Xtr = prep.fit_transform(tr)
    Xte = prep.transform(te)

    det = MyDetector().fit(Xtr)             # labels NOT passed
    pred = det.predict(Xte)
    scores = det.anomaly_score(Xte)

    y = te[contract.LABEL_COL].to_numpy()
    m = contract.binary_metrics(y, pred, scores)
    print(f"[Phase 2] P={m['precision']:.3f} R={m['recall']:.3f} "
          f"F1={m['f1']:.3f} AUC={m.get('auc_roc', float('nan')):.3f} FP={m['fp']}")
    print("[Phase 2] per-attack:",
          {k: round(v["detection_rate"], 3) for k, v in
           contract.per_attack_detection_rate(te[contract.ATTACK_TYPE_COL].to_numpy(), pred).items()})

    os.makedirs(os.path.dirname(contract.UNSUP_BUNDLE), exist_ok=True)
    contract.save_bundle(contract.UNSUP_BUNDLE,
                         contract.Bundle("unsupervised", prep, det, {}))


if __name__ == "__main__":
    # Import under the package name so the classes pickled inside the bundle
    # resolve as team_kit.group_unsupervised_starter.* (loadable from any
    # process), not as __main__.* (which only exists while this script runs).
    from team_kit import group_unsupervised_starter as _m
    _m.main()
