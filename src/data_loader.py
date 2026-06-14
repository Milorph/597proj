"""
Data access layer.

Loads the CIC IoT-DIAD 2024 packet-level and flow-level tables. If the real
CSVs are present under the configured paths they are used directly; otherwise a
schema-faithful synthetic population is generated (see ``data_synth``). The rest
of the pipeline is agnostic to which path was taken.
"""
from __future__ import annotations

import os

import pandas as pd

import config
from src import data_synth


def _real_files_present() -> bool:
    return os.path.exists(config.PACKET_CSV) and os.path.exists(config.FLOW_CSV)


def load_populations(seed: int = config.DEFAULT_SEED,
                     benign_pop: int | None = None,
                     attack_pop_per_type: int | None = None,
                     verbose: bool = True):
    """
    Return (packets_df, flows_df, source) where ``source`` is 'real' or 'synthetic'.

    The synthetic path is fully deterministic given ``seed`` so results are
    reproducible without committing multi-hundred-MB CSVs to the repo.
    """
    if _real_files_present():
        if verbose:
            print(f"[data] Loading REAL dataset:\n  {config.PACKET_CSV}\n  {config.FLOW_CSV}")
        packets = pd.read_csv(config.PACKET_CSV)
        flows = pd.read_csv(config.FLOW_CSV)
        packets = _normalize_labels(packets)
        flows = _normalize_labels(flows)
        return packets, flows, "real"

    if verbose:
        print("[data] Real CSVs not found -> generating schema-faithful synthetic "
              "population (see README 'Data' section).")
    packets, flows = data_synth.generate(seed=seed, benign_pop=benign_pop,
                                         attack_pop_per_type=attack_pop_per_type)
    if verbose:
        print(f"[data] Synthetic population: {len(packets):,} packet rows, "
              f"{len(flows):,} flow-segment rows.")
    return packets, flows, "synthetic"


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map heterogeneous real-data label spellings onto the canonical taxonomy.

    Real CIC files label every row with an attack-type string (or 'Benign').
    We derive a binary ``Label`` column and a canonical ``attack_type`` column.
    This function is intentionally forgiving so it survives minor naming drift
    in the published files.
    """
    if "attack_type" in df.columns and "Label" in df.columns:
        return df

    # Try to find a label-like column.
    cand = None
    for c in ("Label", "label", "Attack", "attack", "Class", "class", "Category"):
        if c in df.columns:
            cand = c
            break
    if cand is None:
        raise ValueError("Could not find a label column in real data; "
                         "expected one of Label/Attack/Class/Category.")

    raw = df[cand].astype(str).str.strip()
    norm = (raw.str.replace(" ", "_", regex=False)
               .str.replace("-", "-", regex=False))

    def to_canonical(v: str) -> str:
        low = v.lower()
        if "benign" in low or low in ("normal", "0"):
            return config.BENIGN_LABEL
        if "ddos" in low and "http" in low:
            return "DDoS-HTTP_Flood"
        if "dos" in low and "http" in low:
            return "DoS-HTTP_Flood"
        if "dns" in low and "spoof" in low:
            return "DNS_Spoofing"
        if "xss" in low:
            return "XSS"
        if "brute" in low:
            return "Brute_Force"
        return v  # leave unknown attack strings as-is

    df = df.copy()
    df["attack_type"] = norm.map(to_canonical)
    df["Label"] = (df["attack_type"] != config.BENIGN_LABEL).astype(int)
    return df
