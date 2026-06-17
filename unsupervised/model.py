# phase 2 model: preprocessing + unsupervised detector

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# columns that are ids/labels, not features
DROP = {"flow_id", "src_ip", "dst_ip", "src_port", "dst_port",
        "timestamp", "segment_index", "Label", "attack_type"}


class Prep:
    def __init__(self):
        self.cols = None
        self.imp = SimpleImputer(strategy="median")
        self.sc = StandardScaler()

    def _num(self, df):
        num = df.select_dtypes("number")
        keep = [c for c in num.columns if c not in DROP]
        return num[keep].replace([np.inf, -np.inf], np.nan)

    def fit_transform(self, df):           # fit on train only
        X = self._num(df)
        self.cols = list(X.columns)
        return self.sc.fit_transform(self.imp.fit_transform(X))

    def transform(self, df):
        X = df.reindex(columns=self.cols).replace([np.inf, -np.inf], np.nan)
        return self.sc.transform(self.imp.transform(X))


class Detector:
    # autoencoder reconstruction error + isolation forest + kmeans distance, averaged
    def __init__(self, budget=0.20, seed=42):
        self.budget = budget          # flag top 20% as suspicious (wide net)
        self.seed = seed

    def fit(self, X):                  # NOTE: no labels here
        self.ae = MLPRegressor(hidden_layer_sizes=(32, 12, 32),
                               max_iter=60, random_state=self.seed)
        self.ae.fit(X, X)              # learn to rebuild benign-ish traffic
        self.iso = IsolationForest(n_estimators=200, random_state=self.seed,
                                   n_jobs=-1).fit(X)
        r = self._recon(X)
        self.km = KMeans(8, n_init=10, random_state=self.seed).fit(np.c_[X, r])
        s = self._score(X)
        self.thr = float(np.quantile(s, 1 - self.budget))
        return self

    def _recon(self, X):               # per-row reconstruction error
        return ((X - self.ae.predict(X)) ** 2).mean(1)

    def _z(self, a):
        return (a - a.mean()) / (a.std() + 1e-9)

    def _score(self, X):               # combined anomaly score (higher = worse)
        r = self._recon(X)
        iso = -self.iso.score_samples(X)
        d = self.km.transform(np.c_[X, r]).min(1)
        return (self._z(r) + self._z(iso) + self._z(d)) / 3

    def score(self, X):
        return self._score(X)

    def predict(self, X):              # 1 = alert
        return (self._score(X) >= self.thr).astype(int)
