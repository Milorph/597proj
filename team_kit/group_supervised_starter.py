"""
GROUP: SUPERVISED  (Phase 3 -- signature-based IDS)
==================================================
You own this file + your own preprocessing. You implement a SignatureClassifier.
You use labels for training, but NEVER at test time.

Run standalone, no other group needed:
    python -m team_kit.group_supervised_starter

Deliverable to the cascade group: team_kit/artifacts/supervised.joblib
(a fitted preprocessor + classifier bundle, plus a tuned threshold in `extra`).
A RandomForest baseline is provided -- swap for XGBoost / tune to improve it.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from team_kit import contract


# ---- YOUR preprocessing (you own this; ships bundled with your model) ------- #
class FlowPreprocessor:
    feature_names: list = []

    def __init__(self):
        self.imp = SimpleImputer(strategy="median")
        self.sc = StandardScaler()

    def _mat(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = {contract.LABEL_COL, contract.ATTACK_TYPE_COL, contract.FLOW_ID_COL,
                "src_ip", "dst_ip", "segment_index", "n_segments"}
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


# ---- YOUR model: implement contract.SignatureClassifier --------------------- #
class MyClassifier:
    """Baseline RandomForest. TODO: try XGBoost, tune depth/estimators."""
    def __init__(self, seed=42):
        self.clf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def tune_threshold(self, Xv, yv):
        p = self.predict_proba(Xv)
        grid = np.linspace(0.05, 0.95, 19)
        return float(max(grid, key=lambda t: f1_score(yv, (p >= t).astype(int), zero_division=0)))


def main():
    df = contract.get_flow_sample(seed=42)
    tr, tmp = train_test_split(df, test_size=0.3, random_state=42,
                               stratify=df[contract.ATTACK_TYPE_COL])
    val, te = train_test_split(tmp, test_size=0.5, random_state=42,
                               stratify=tmp[contract.ATTACK_TYPE_COL])

    prep = FlowPreprocessor()
    Xtr, Xval, Xte = prep.fit_transform(tr), prep.transform(val), prep.transform(te)
    ytr = tr[contract.LABEL_COL].to_numpy()
    yval = val[contract.LABEL_COL].to_numpy()
    yte = te[contract.LABEL_COL].to_numpy()

    clf = MyClassifier().fit(Xtr, ytr)
    thr = clf.tune_threshold(Xval, yval)
    m = contract.binary_metrics(yte, clf.predict(Xte, thr), clf.predict_proba(Xte))
    print(f"[Phase 3] F1={m['f1']:.3f} AUC={m['auc_roc']:.3f} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} thr={thr:.2f}")

    os.makedirs(os.path.dirname(contract.SUP_BUNDLE), exist_ok=True)
    contract.save_bundle(contract.SUP_BUNDLE,
                         contract.Bundle("supervised", prep, clf, {"threshold": thr}))


if __name__ == "__main__":
    # Import under the package name so the classes pickled inside the bundle
    # resolve from any process (not as __main__.*).
    from team_kit import group_supervised_starter as _m
    _m.main()
