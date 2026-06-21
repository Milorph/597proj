# models for both sections (kept here so saved bundles load cleanly later)

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGB = False

# columns to drop (ids/labels, not features)
DROP_PKT = {"flow_id", "src_ip", "dst_ip", "src_port", "dst_port",
            "timestamp", "segment_index", "Label", "attack_type"}
DROP_FLOW = {"flow_id", "src_ip", "dst_ip", "src_port", "dst_port",
             "segment_index", "n_segments", "Label", "attack_type"}


class Prep:
    # numeric cols only, median fill, scale. fit on train, apply to test
    def __init__(self, drop):
        self.drop = set(drop)
        self.cols = None
        self.imp = SimpleImputer(strategy="median")
        self.sc = StandardScaler()

    def _num(self, df):
        num = df.select_dtypes("number")
        keep = [c for c in num.columns if c not in self.drop]
        return num[keep].replace([np.inf, -np.inf], np.nan)

    def fit_transform(self, df):
        X = self._num(df)
        self.cols = list(X.columns)
        return self.sc.fit_transform(self.imp.fit_transform(X))

    def transform(self, df):
        X = df.reindex(columns=self.cols).replace([np.inf, -np.inf], np.nan)
        return self.sc.transform(self.imp.transform(X))


# ---- unsupervised (phase 2) ----
class Detector:
    # autoencoder reconstruction error + isolation forest + kmeans distance, averaged
    def __init__(self, budget=0.20, seed=42):
        self.budget = budget          # flag top 20% as suspicious (wide net)
        self.seed = seed

    def fit(self, X):                  # no labels here
        self.ae = MLPRegressor(hidden_layer_sizes=(32, 12, 32),
                               max_iter=60, random_state=self.seed)
        self.ae.fit(X, X)
        self.iso = IsolationForest(n_estimators=200, random_state=self.seed,
                                   n_jobs=-1).fit(X)
        r = self._recon(X)
        self.km = KMeans(8, n_init=10, random_state=self.seed).fit(np.c_[X, r])
        self.thr = float(np.quantile(self._score(X), 1 - self.budget))
        return self

    def _recon(self, X):
        return ((X - self.ae.predict(X)) ** 2).mean(1)

    def _z(self, a):
        return (a - a.mean()) / (a.std() + 1e-9)

    def _score(self, X):               # higher = more anomalous
        r = self._recon(X)
        iso = -self.iso.score_samples(X)
        d = self.km.transform(np.c_[X, r]).min(1)
        return (self._z(r) + self._z(iso) + self._z(d)) / 3

    def score(self, X):
        return self._score(X)

    def predict(self, X):              # 1 = alert
        return (self._score(X) >= self.thr).astype(int)


# ---- supervised (phase 3) ----
class Classifier:
    def __init__(self, seed=42):
        if HAS_XGB:
            self.clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                     subsample=0.9, colsample_bytree=0.9,
                                     eval_metric="logloss", random_state=seed,
                                     n_jobs=-1, tree_method="hist")
        else:
            self.clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                              random_state=seed, n_jobs=-1)

    def fit(self, X, y):
        if HAS_XGB:                    # weight rare attacks up
            pos = max(1, int((y == 1).sum()))
            self.clf.set_params(scale_pos_weight=(y == 0).sum() / pos)
        self.clf.fit(X, y)
        return self

    def proba(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X, t=0.5):
        return (self.proba(X) >= t).astype(int)

    def tune(self, Xv, yv):            # best threshold on validation F1
        p = self.proba(Xv)
        grid = np.linspace(0.05, 0.95, 19)
        return float(max(grid, key=lambda t: f1_score(yv, (p >= t).astype(int),
                                                       zero_division=0)))
