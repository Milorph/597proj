"""
Phase 2 -- Unsupervised anomaly detection (anomaly-based IDS).

We combine three complementary unsupervised signals into one anomaly score:

  1. **Autoencoder reconstruction error** (MLP autoencoder). Trained to
     reconstruct its input; because benign traffic dominates, attacks live off
     the learned benign manifold and reconstruct poorly.
  2. **Isolation Forest** path-length score -- isolates globally rare points.
  3. **K-means distance-to-centroid** on the AE-augmented feature space --
     captures local density. We deliberately *over-cluster* (k=8) and read off
     each cluster's anomaly tendency, which also answers the brief's question
     about how DoS/DDoS sit in the cluster structure.

No labels are used in ``fit``. The operational threshold is set from the known
class imbalance (attacks ~2-3% => flag the top-`contamination` fraction of
scores), which is label-free and justified by the dataset design.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

import config


def _zscore(a: np.ndarray) -> np.ndarray:
    mu, sd = a.mean(), a.std()
    return (a - mu) / (sd + 1e-9)


class UnsupervisedDetector:
    """Autoencoder + Isolation Forest + K-means anomaly ensemble."""

    def __init__(self, seed: int = config.DEFAULT_SEED,
                 ae_hidden=config.AE_HIDDEN, ae_max_iter=config.AE_MAX_ITER,
                 k=config.KMEANS_K, iso_estimators=config.ISO_FOREST_ESTIMATORS,
                 contamination: float = 0.03):
        self.seed = seed
        self.ae_hidden = ae_hidden
        self.ae_max_iter = ae_max_iter
        self.k = k
        self.iso_estimators = iso_estimators
        self.contamination = contamination

        self.ae: MLPRegressor | None = None
        self.iso: IsolationForest | None = None
        self.kmeans: KMeans | None = None
        self.threshold_: float | None = None
        self.cluster_anomaly_: dict | None = None
        self._fit_scores = None

    # -- training ----------------------------------------------------------
    def fit(self, X: np.ndarray) -> "UnsupervisedDetector":
        rng = self.seed

        # 1) Autoencoder: regress X onto itself through a bottleneck.
        self.ae = MLPRegressor(hidden_layer_sizes=self.ae_hidden,
                               activation="relu", solver="adam",
                               max_iter=self.ae_max_iter, random_state=rng,
                               early_stopping=False, batch_size=256)
        self.ae.fit(X, X)
        recon = self._recon_error(X)

        # 2) Isolation Forest on the raw features.
        self.iso = IsolationForest(n_estimators=self.iso_estimators,
                                   contamination=self.contamination,
                                   random_state=rng, n_jobs=-1)
        self.iso.fit(X)

        # 3) K-means on AE-augmented space (features + recon error).
        Xa = np.hstack([X, recon.reshape(-1, 1)])
        self.kmeans = KMeans(n_clusters=self.k, random_state=rng, n_init=10)
        self.kmeans.fit(Xa)

        # Map each cluster to an anomaly tendency = mean recon error of members
        # (label-free). Clusters above the global median are "anomaly-leaning".
        labels = self.kmeans.labels_
        med = np.median(recon)
        self.cluster_anomaly_ = {
            c: float(recon[labels == c].mean()) for c in range(self.k)}

        self._fit_scores = self._combined_score(X, recon)
        # Operational threshold: top-`contamination` fraction are anomalies.
        self.threshold_ = float(np.quantile(self._fit_scores, 1 - self.contamination))
        return self

    def set_alert_budget(self, alert_budget: float) -> float:
        """
        Re-set the operational threshold to flag the top ``alert_budget``
        fraction of training scores. An anomaly-based IDS deliberately casts a
        wide net (high recall, tolerant of false positives); the downstream
        supervised stage then removes the false positives. Returns the new
        threshold.
        """
        self.threshold_ = float(np.quantile(self._fit_scores, 1 - alert_budget))
        return self.threshold_

    # -- scoring -----------------------------------------------------------
    def _recon_error(self, X: np.ndarray) -> np.ndarray:
        pred = self.ae.predict(X)
        return np.mean((X - pred) ** 2, axis=1)

    def _combined_score(self, X: np.ndarray, recon: np.ndarray | None = None) -> np.ndarray:
        if recon is None:
            recon = self._recon_error(X)
        iso_score = -self.iso.score_samples(X)             # higher = more anomalous
        Xa = np.hstack([X, recon.reshape(-1, 1)])
        dists = self.kmeans.transform(Xa).min(axis=1)      # distance to nearest centroid
        return (_zscore(recon) + _zscore(iso_score) + _zscore(dists)) / 3.0

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return self._combined_score(X)

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        thr = self.threshold_ if threshold is None else threshold
        return (self.anomaly_score(X) >= thr).astype(int)

    # -- component scores (for ablation / analysis) ------------------------
    def component_scores(self, X: np.ndarray) -> dict:
        recon = self._recon_error(X)
        iso_score = -self.iso.score_samples(X)
        Xa = np.hstack([X, recon.reshape(-1, 1)])
        dists = self.kmeans.transform(Xa).min(axis=1)
        return {"autoencoder": recon, "isolation_forest": iso_score,
                "kmeans_distance": dists, "combined": self._combined_score(X, recon)}

    def cluster_assignments(self, X: np.ndarray) -> np.ndarray:
        recon = self._recon_error(X)
        Xa = np.hstack([X, recon.reshape(-1, 1)])
        return self.kmeans.predict(Xa)


def select_threshold_by_fpr(scores: np.ndarray, y_true: np.ndarray,
                            target_fpr: float = config.TARGET_FPR) -> float:
    """
    Label-aware threshold helper used only for *analysis* (Task 2.2): the
    largest score cut-off whose benign false-positive rate stays <= target_fpr.
    Demonstrates the FP/FN trade-off; the operational detector still uses the
    label-free contamination threshold.
    """
    benign_scores = np.sort(scores[y_true == 0])
    idx = int((1 - target_fpr) * len(benign_scores))
    idx = min(idx, len(benign_scores) - 1)
    return float(benign_scores[idx])
