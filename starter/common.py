# shared helpers: load real csvs, sample, aggregate flows, metrics

import os
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, roc_auc_score)

BENIGN = "Benign"
ATTACK_TYPES = ["DDoS-HTTP_Flood", "DoS-HTTP_Flood", "DNS_Spoofing", "XSS", "Brute_Force"]

DATA_DIR = os.environ.get("IDS_DATA_DIR", "data")
PACKET_DIR = os.environ.get("IDS_PACKET_DIR", os.path.join(DATA_DIR, "packet"))
FLOW_DIR = os.environ.get("IDS_FLOW_DIR", os.path.join(DATA_DIR, "flow"))
MAX_ROWS = int(os.environ["IDS_MAX_ROWS_PER_FILE"]) if os.environ.get("IDS_MAX_ROWS_PER_FILE") else None


# label from folder/file name
def to_canonical(text):
    s = str(text).lower()
    if "benign" in s or "normal" in s:
        return BENIGN
    if "ddos" in s and "http" in s:
        return "DDoS-HTTP_Flood"
    if "dos" in s and "http" in s:
        return "DoS-HTTP_Flood"
    if ("dns" in s and "spoof" in s) or "dnsspoof" in s:
        return "DNS_Spoofing"
    if "xss" in s:
        return "XSS"
    if "brute" in s:
        return "Brute_Force"
    return None                       # mirai/recon -> ignored


# build a canonical, direction-independent flow_id
def _norm(c):
    return str(c).lower().replace(" ", "").replace("_", "").replace("-", "")

_ALIAS = {"src_ip": {"srcip", "sourceip", "ipsrc", "saddr"},
          "dst_ip": {"dstip", "destip", "destinationip", "ipdst", "daddr"},
          "src_port": {"srcport", "sourceport", "sport"},
          "dst_port": {"dstport", "destport", "destinationport", "dport"}}

def _port(s):
    return pd.to_numeric(s, errors="coerce").fillna(-1).astype("int64").astype(str)

def _flow_id(df):
    df = df.copy()
    for canon, aliases in _ALIAS.items():
        if canon not in df.columns:
            for c in df.columns:
                if _norm(c) in aliases:
                    df = df.rename(columns={c: canon})
                    break
    ip1 = df["src_ip"].astype("string").fillna("na").str.strip() + ":" + _port(df["src_port"])
    ip2 = df["dst_ip"].astype("string").fillna("na").str.strip() + ":" + _port(df["dst_port"])
    ip1, ip2 = ip1.fillna("na:-1"), ip2.fillna("na:-1")
    swap = (ip1 > ip2).fillna(False).to_numpy()      # sort the two endpoints
    df["flow_id"] = np.where(swap, ip2.str.cat(ip1, sep="-"), ip1.str.cat(ip2, sep="-"))
    return df


def _read_dir(path):
    files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no csv files under {path} (put your data there)")
    frames = []
    for f in files:
        cls = to_canonical(os.path.relpath(f, path))
        if cls is None:
            continue                                  # skip mirai/recon/etc
        d = pd.read_csv(f, low_memory=False, nrows=MAX_ROWS, on_bad_lines="skip")
        d.columns = [str(c).strip() for c in d.columns]
        d = d.replace([np.inf, -np.inf], np.nan)
        d["attack_type"] = cls
        d["Label"] = int(cls != BENIGN)
        frames.append(d)
        print(f"  {os.path.relpath(f, path)} -> {cls} ({len(d):,})")
    return _flow_id(pd.concat(frames, ignore_index=True, sort=False))


def load_packets():
    return _read_dir(PACKET_DIR)

def load_flows():
    return _read_dir(FLOW_DIR)


# task 1.1 sampler: 200k benign + 2-3% attack spread evenly across the 5 types
def sample_dataset(pop, seed=None, n_benign=200_000, attack_min=4_000, attack_max=6_200):
    rng = np.random.default_rng(seed)
    benign = pop[pop["Label"] == 0]
    if len(benign) == 0:
        raise ValueError("no benign rows")
    if len(benign) < n_benign:                        # real data may have fewer (esp flows)
        f = len(benign) / n_benign
        attack_min, attack_max = max(5, int(attack_min * f)), max(5, int(attack_max * f))
        print(f"  capping benign to {len(benign):,}, attack -> {attack_min}-{attack_max}")
        n_benign = len(benign)

    total = int(rng.integers(attack_min, attack_max + 1))
    raw = (total / 5) * rng.uniform(0.85, 1.15, 5)    # +/-15% jitter
    counts = np.maximum(1, np.floor(raw / raw.sum() * total).astype(int))
    while counts.sum() > total and counts.max() > 1:
        counts[counts.argmax()] -= 1
    while counts.sum() < total:
        counts[rng.integers(0, 5)] += 1

    parts = [benign.sample(n_benign, random_state=int(rng.integers(0, 2**31)))]
    for atk, c in zip(ATTACK_TYPES, counts):
        pool = pop[pop["attack_type"] == atk]
        parts.append(pool.sample(min(int(c), len(pool)), random_state=int(rng.integers(0, 2**31))))
    out = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=int(rng.integers(0, 2**31)))
    n_atk = int(out["Label"].sum())
    print(f"  sample: {len(out):,} rows, {n_atk:,} attack ({n_atk/len(out):.2%})")
    return out.reset_index(drop=True)


# group 2-min flow segments into one record per flow
def _agg(col):
    n = col.lower()
    if "/s" in n or "mean" in n or "avg" in n or "std" in n or "rate" in n or "ratio" in n:
        return "mean"
    if "max" in n:
        return "max"
    if "min" in n:
        return "min"
    if any(k in n for k in ("tot", "count", "duration", "bytes", "packet", "pkts", "flag", "len")):
        return "sum"
    return "mean"

def aggregate_flows(flow_df):
    skip = {"flow_id", "Label", "attack_type", "src_ip", "dst_ip", "src_port", "dst_port"}
    spec = {c: _agg(c) for c in flow_df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(flow_df[c])}
    g = flow_df.groupby("flow_id", sort=False)
    out = g.agg(spec)
    out["Label"] = g["Label"].max()
    out["attack_type"] = g["attack_type"].agg(lambda s: s.value_counts().index[0])
    out["n_segments"] = g.size()
    out = out.reset_index()
    print(f"  {len(flow_df):,} segments -> {len(out):,} flows")
    return out


# metrics
def binary_metrics(y, pred, scores=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {"precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred, zero_division=0),
         "f1": f1_score(y, pred, zero_division=0),
         "accuracy": accuracy_score(y, pred),
         "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
         "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
         "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
    if scores is not None:
        try:
            m["auc_roc"] = float(roc_auc_score(y, scores))
        except ValueError:
            m["auc_roc"] = float("nan")
    return m

def per_attack_detection_rate(y_multi, pred):
    out = {}
    for atk in ATTACK_TYPES:
        mask = (y_multi == atk)
        n = int(mask.sum())
        det = int(pred[mask].sum()) if n else 0
        out[atk] = {"detected": det, "total": n,
                    "detection_rate": (det / n) if n else float("nan")}
    return out
