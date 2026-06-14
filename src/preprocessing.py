"""
Task 1.2 -- Preprocessing.

A small, transparent preprocessing pipeline. Every choice is documented inline
and surfaced in the report:

  1. Drop identifier / leakage columns (IPs, ports as raw strings, timestamps,
     flow_id) -- these memorise hosts rather than generalise to attack behaviour.
  2. Coerce to numeric and impute missing values (median) -- network captures
     frequently contain NaN/Inf for rate features on 1-packet flows.
  3. Replace +/-Inf (e.g. bytes-per-second on a 0-duration flow) with NaN then
     impute.
  4. Clip extreme outliers to robust quantiles (1st/99th) -- floods produce
     legitimate but enormous values that otherwise dominate scaling.
  5. Standard-scale features (zero mean, unit variance) -- required by
     distance-based (k-means) and gradient-based (autoencoder) methods.

The fitted transformer is reusable so train/test and packet/flow stages share
exactly the same transformation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import config

# Columns that identify hosts/sessions rather than describe behaviour. Kept out
# of the model matrix to avoid memorising IPs/ports (label leakage).
ID_COLS = ["flow_id", "src_ip", "dst_ip", "timestamp", "segment_index"]
LABEL_COLS = ["Label", "attack_type"]


class Preprocessor:
    """Fit-once / apply-many preprocessing transformer."""

    def __init__(self, clip_quantiles=(0.01, 0.99)):
        self.clip_quantiles = clip_quantiles
        self.feature_cols_: list[str] | None = None
        self.lo_ = None
        self.hi_ = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    # -- helpers -----------------------------------------------------------
    def _select_features(self, df: pd.DataFrame) -> list[str]:
        drop = set(ID_COLS + LABEL_COLS)
        cols = []
        for c in df.columns:
            if c in drop:
                continue
            # Keep numeric-coercible columns only.
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
            else:
                # Ports may arrive as strings; coerce if it works.
                coerced = pd.to_numeric(df[c], errors="coerce")
                if coerced.notna().mean() > 0.5:
                    cols.append(c)
        return cols

    def _clean_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols_].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        return X

    # -- API ---------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.feature_cols_ = self._select_features(df)
        X = self._clean_matrix(df)
        # Robust clip bounds from the (pre-impute) distribution.
        self.lo_ = X.quantile(self.clip_quantiles[0])
        self.hi_ = X.quantile(self.clip_quantiles[1])
        X = X.clip(self.lo_, self.hi_, axis=1)
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_cols_ is None:
            raise RuntimeError("Preprocessor must be fit before transform().")
        X = self._clean_matrix(df)
        X = X.clip(self.lo_, self.hi_, axis=1)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return X

    @property
    def feature_names(self) -> list[str]:
        return list(self.feature_cols_ or [])


def get_labels(df: pd.DataFrame):
    """Return (y_binary, y_multiclass) arrays from a labelled frame."""
    y_bin = df["Label"].to_numpy().astype(int)
    y_multi = df["attack_type"].to_numpy()
    return y_bin, y_multi
