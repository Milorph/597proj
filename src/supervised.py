"""
Phase 3 -- Supervised classifier (signature-based IDS).

A gradient-boosted tree model (XGBoost when available, else sklearn's
HistGradientBoosting) trained on flow-level features to re-evaluate the alerts
raised in Phase 2. Gradient-boosted trees are a strong, well-understood baseline
for tabular network-flow data: they handle mixed feature scales, are robust to
irrelevant features, and train quickly on hundreds of thousands of rows.

The model is trained on the flow-level sample and can optionally consume the
flow-level *anomaly score* produced by the Phase-2 detector applied to flows
(Task 3.1), so its marginal contribution can be measured (with/without).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

import config

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:                                            # pragma: no cover
    _HAS_XGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier


class SupervisedIDS:
    """Binary benign-vs-attack classifier for the flow stage."""

    def __init__(self, seed: int = config.DEFAULT_SEED, scale_pos_weight: float | None = None):
        self.seed = seed
        self.scale_pos_weight = scale_pos_weight
        self.model = None
        self.backend = "xgboost" if _HAS_XGB else "hist_gbdt"

    def fit(self, X, y, X_val=None, y_val=None):
        if _HAS_XGB:
            spw = self.scale_pos_weight
            if spw is None:
                pos = max(1, int(np.sum(y == 1)))
                neg = int(np.sum(y == 0))
                spw = neg / pos                              # counter class imbalance
            self.model = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9,
                scale_pos_weight=spw, eval_metric="logloss",
                random_state=self.seed, n_jobs=-1, tree_method="hist")
            fit_kw = {}
            if X_val is not None:
                fit_kw["eval_set"] = [(X_val, y_val)]
                fit_kw["verbose"] = False
            self.model.fit(X, y, **fit_kw)
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.1, max_depth=6,
                class_weight="balanced", random_state=self.seed)
            self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def tune_threshold(self, X_val, y_val) -> float:
        """Pick the probability cut-off that maximises validation F1."""
        proba = self.predict_proba(X_val)
        grid = np.linspace(0.05, 0.95, 19)
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(y_val, (proba >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return float(best_t)

    def feature_importance(self, feature_names) -> dict:
        if _HAS_XGB:
            imp = self.model.feature_importances_
        else:
            # HistGBDT has no native importances; fall back to permutation-free zeros.
            imp = getattr(self.model, "feature_importances_", np.zeros(len(feature_names)))
        return dict(sorted(zip(feature_names, map(float, imp)),
                           key=lambda kv: kv[1], reverse=True))
