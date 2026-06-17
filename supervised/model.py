# phase 3 model: preprocessing + supervised classifier

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:                       # fallback if xgboost missing
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGB = False

# ids/labels, not features
DROP = {"flow_id", "src_ip", "dst_ip", "src_port", "dst_port",
        "segment_index", "n_segments", "Label", "attack_type"}


class Prep:
    def __init__(self):
        self.cols = None
        self.imp = SimpleImputer(strategy="median")
        self.sc = StandardScaler()

    def _num(self, df):
        num = df.select_dtypes("number")
        keep = [c for c in num.columns if c not in DROP]
        return num[keep].replace([np.inf, -np.inf], np.nan)

    def fit_transform(self, df):
        X = self._num(df)
        self.cols = list(X.columns)
        return self.sc.fit_transform(self.imp.fit_transform(X))

    def transform(self, df):
        X = df.reindex(columns=self.cols).replace([np.inf, -np.inf], np.nan)
        return self.sc.transform(self.imp.transform(X))


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
        if HAS_XGB:                      # weight rare attacks up
            pos = max(1, int((y == 1).sum()))
            self.clf.set_params(scale_pos_weight=(y == 0).sum() / pos)
        self.clf.fit(X, y)
        return self

    def proba(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X, t=0.5):
        return (self.proba(X) >= t).astype(int)

    def tune(self, Xv, yv):              # pick threshold with best val F1
        p = self.proba(Xv)
        grid = np.linspace(0.05, 0.95, 19)
        return float(max(grid, key=lambda t: f1_score(yv, (p >= t).astype(int),
                                                       zero_division=0)))
