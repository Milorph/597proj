"""
team_kit.mocks  --  stand-ins so a group can run END-TO-END alone.

The CASCADE group uses MockDetector + MockClassifier to build and test the
two-stage logic before the other two groups deliver anything. At integration
they swap these for the real bundles (contract.load_bundle).

The UNSUPERVISED / SUPERVISED groups generally don't need mocks (they train on
real synthetic data), but MockTransformer is here if they want a quick stub.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from team_kit import contract


class MockTransformer:
    """Trivial preprocessing: keep numeric columns, fill NaN with 0."""
    def __init__(self):
        self.feature_names = []

    def _num(self, df):
        num = df.select_dtypes(include="number")
        drop = {contract.LABEL_COL}
        return num[[c for c in num.columns if c not in drop]]

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._num(df)
        self.feature_names = list(X.columns)
        return X.fillna(0.0).to_numpy()

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.feature_names, fill_value=0.0)
        return X.fillna(0.0).to_numpy()


class MockDetector:
    """Pretend anomaly detector: scores ~ first feature + noise (deterministic)."""
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._thr = 0.0

    def fit(self, X: np.ndarray) -> "MockDetector":
        s = self._raw(X)
        self._thr = float(np.quantile(s, 0.80))   # flag top 20%
        return self

    def _raw(self, X):
        base = X[:, 0] if X.shape[1] else np.zeros(len(X))
        return base + self.rng.standard_normal(len(X)) * 0.1

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return self._raw(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.anomaly_score(X) >= self._thr).astype(int)


class MockClassifier:
    """Pretend supervised classifier: P(attack) ~ sigmoid(first feature)."""
    def __init__(self, seed: int = 1):
        self.rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MockClassifier":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X[:, 0] if X.shape[1] else np.zeros(len(X))
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
