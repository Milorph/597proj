"""
Data access layer.

Loads the CIC IoT-DIAD 2024 packet-level and flow-level tables. Three input
modes are supported, tried in order:

  1. **Single CSVs** -- ``data/packet_level.csv`` + ``data/flow_level.csv``.
  2. **Folders of raw CIC files** -- ``data/packet/`` + ``data/flow/``. Every
     ``.csv`` under the folder is read and its attack label is inferred from the
     sub-folder/file name (Benign, BruteForce, DDoS, DoS, Spoofing, Web-Based,
     ...), so you can drop the downloaded files straight in without merging.
  3. **Synthetic fallback** -- a schema-faithful generator (``data_synth``) when
     no real data is present, so the pipeline always runs.

For real data the loader also (a) strips whitespace from column names,
(b) builds a canonical ``flow_id`` from the src/dst IP + port columns when one
is not already present, and (c) derives binary ``Label`` + canonical
``attack_type`` columns. The rest of the pipeline is agnostic to the source.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import config
from src import data_synth


# --------------------------------------------------------------------------- #
# Label / class inference
# --------------------------------------------------------------------------- #
def to_canonical(text: str):
    """
    Map an arbitrary label value *or file path* onto the canonical taxonomy.
    Returns ``None`` if it matches none of the project's classes (e.g. the
    Mirai / Recon folders we don't use).
    """
    low = str(text).lower()
    if "benign" in low or "normal" in low:
        return config.BENIGN_LABEL
    if "ddos" in low and "http" in low:
        return "DDoS-HTTP_Flood"
    if "dos" in low and "http" in low:          # checked after ddos
        return "DoS-HTTP_Flood"
    if ("dns" in low and "spoof" in low) or "dnsspoof" in low:
        return "DNS_Spoofing"
    if "xss" in low:
        return "XSS"
    if "brute" in low:
        return "Brute_Force"
    return None


# --------------------------------------------------------------------------- #
# Identifier handling (build flow_id for real data)
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    return str(name).lower().replace(" ", "").replace("_", "").replace("-", "")


_ID_ALIASES = {
    "flow_id": {"flowid"},
    "src_ip": {"srcip", "sourceip", "ipsrc", "saddr", "sourceaddress"},
    "dst_ip": {"dstip", "destip", "destinationip", "ipdst", "daddr", "destinationaddress"},
    "src_port": {"srcport", "sourceport", "sport"},
    "dst_port": {"dstport", "destport", "destinationport", "dport"},
}


def _find_col(df: pd.DataFrame, canonical: str):
    aliases = _ID_ALIASES[canonical]
    for c in df.columns:
        if _norm(c) in aliases:
            return c
    return None


def _port_str(s: pd.Series) -> pd.Series:
    """Normalise a port column to a clean integer string ('80' not '80.0')."""
    return pd.to_numeric(s, errors="coerce").fillna(-1).astype("int64").astype(str)


def _ensure_flow_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee a ``flow_id`` column and canonical id columns where possible.

    For real CIC data we build a **canonical, direction-independent** flow_id
    from the 4-tuple (src IP+port, dst IP+port) by ordering the two endpoints,
    so a packet seen in either direction maps to the same flow as its
    (bi-directional) CICFlowMeter flow record. We deliberately ignore any
    pre-existing ``Flow ID`` column, because CICFlowMeter's includes the
    protocol (5-tuple) and a fixed direction, which would not match the packet
    side. This matches the brief's definition: srcIP-dstIP-sport-dport.
    """
    df = df.copy()
    # Rename detected identifier columns to canonical names (non-destructive).
    for canon in ("flow_id", "src_ip", "dst_ip", "src_port", "dst_port"):
        if canon in df.columns:
            continue
        found = _find_col(df, canon)
        if found is not None:
            df = df.rename(columns={found: canon})

    have4 = all(c in df.columns for c in ("src_ip", "dst_ip", "src_port", "dst_port"))
    if have4:
        ep1 = df["src_ip"].astype(str).str.strip() + ":" + _port_str(df["src_port"])
        ep2 = df["dst_ip"].astype(str).str.strip() + ":" + _port_str(df["dst_port"])
        a, b = ep1.to_numpy(), ep2.to_numpy()
        df["flow_id"] = np.where(a <= b, a + "-" + b, b + "-" + a)
    elif "flow_id" not in df.columns:
        raise ValueError(
            "Cannot build flow_id: need a 'Flow ID' column or all of "
            "src/dst IP + src/dst port. Found columns: "
            + ", ".join(map(str, df.columns[:30])))
    return df


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def _read_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, nrows=config.MAX_ROWS_PER_FILE,
                     on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _read_dir(path: str, kind: str) -> pd.DataFrame:
    """Read+concat every CSV under ``path``; label each row from its file path."""
    files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files under {path}")
    frames, skipped = [], []
    for f in files:
        cls = to_canonical(os.path.relpath(f, path))      # infer from sub-path
        if cls is None:
            skipped.append(os.path.basename(f))
            continue
        d = _read_one_csv(f)
        d["attack_type"] = cls
        d["Label"] = int(cls != config.BENIGN_LABEL)
        frames.append(d)
        print(f"[data]   {kind}: {os.path.relpath(f, path)} -> {cls}  ({len(d):,} rows)")
    if skipped:
        print(f"[data]   {kind}: skipped {len(skipped)} unrecognised file(s): "
              f"{', '.join(skipped[:6])}{' ...' if len(skipped) > 6 else ''}")
    if not frames:
        raise ValueError(f"None of the CSVs under {path} matched the five attack "
                         f"categories or Benign.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    return _ensure_flow_id(df)


def _load_real(kind: str, csv_path: str, dir_path: str):
    """Return a labelled real-data DataFrame from a CSV or a folder, or None."""
    if os.path.exists(csv_path):
        print(f"[data] Loading REAL {kind} CSV: {csv_path}")
        df = _read_one_csv(csv_path)
        df = _normalize_labels(df)
        return _ensure_flow_id(df)
    if os.path.isdir(dir_path) and glob.glob(os.path.join(dir_path, "**", "*.csv"), recursive=True):
        print(f"[data] Loading REAL {kind} folder: {dir_path}")
        return _read_dir(dir_path, kind)
    return None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_populations(seed: int = config.DEFAULT_SEED,
                     benign_pop: int | None = None,
                     attack_pop_per_type: int | None = None,
                     verbose: bool = True):
    """
    Return (packets_df, flows_df, source) with ``source`` in {'real','synthetic'}.

    Real data is used when *both* a packet and a flow source are found; otherwise
    the deterministic synthetic population is generated.
    """
    packets = _load_real("packet", config.PACKET_CSV, config.PACKET_DIR)
    flows = _load_real("flow", config.FLOW_CSV, config.FLOW_DIR)
    if packets is not None and flows is not None:
        if verbose:
            print(f"[data] REAL dataset: {len(packets):,} packet rows, "
                  f"{len(flows):,} flow rows.")
        return packets, flows, "real"
    if packets is not None or flows is not None:
        print("[data] WARNING: found only one of packet/flow real data; need BOTH. "
              "Falling back to synthetic. (Check data/packet and data/flow.)")

    if verbose:
        print("[data] Real data not found -> generating schema-faithful synthetic "
              "population (see README 'Data' section).")
    packets, flows = data_synth.generate(seed=seed, benign_pop=benign_pop,
                                         attack_pop_per_type=attack_pop_per_type)
    if verbose:
        print(f"[data] Synthetic population: {len(packets):,} packet rows, "
              f"{len(flows):,} flow-segment rows.")
    return packets, flows, "synthetic"


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive ``Label`` + ``attack_type`` from whatever label column a single real
    CSV carries (used for mode 1; the folder reader labels from the path).
    """
    if "attack_type" in df.columns and "Label" in df.columns:
        return df
    cand = None
    for c in df.columns:
        if _norm(c) in ("label", "attack", "class", "category", "type"):
            cand = c
            break
    if cand is None:
        raise ValueError("Could not find a label column in the real CSV; expected "
                         "one of Label/Attack/Class/Category, or use the folder "
                         "input mode (data/packet, data/flow).")
    df = df.copy()
    mapped = df[cand].map(to_canonical)
    # Unknown strings: keep them as raw attack types (still counted as attacks).
    df["attack_type"] = mapped.where(mapped.notna(), df[cand].astype(str))
    df["Label"] = (df["attack_type"] != config.BENIGN_LABEL).astype(int)
    return df
