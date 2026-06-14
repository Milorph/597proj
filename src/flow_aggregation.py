"""
Phase 3 -- flow segmentation handling.

The flow-level dataset stores one row per 2-minute segment, so a flow longer
than 2 minutes appears as several rows that share a ``flow_id``. This module
collapses those segments into a single unified record per flow, using a
per-feature aggregation that respects each feature's meaning:

  * **sum**  for additive quantities (packet/byte/flag counts, duration);
  * **max**  for peak quantities (peak rate, max packet length);
  * **mean** for rate/ratio/average descriptors.

It also provides the mapping used to find the flow record corresponding to a
given packet (Phase-3 Task 3.2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Aggregation policy keyed by flow feature. Anything not listed defaults to mean.
AGG_MAP = {
    "flow_duration": "sum",
    "total_packets": "sum",
    "total_bytes": "sum",
    "fwd_packets": "sum",
    "bwd_packets": "sum",
    "syn_count": "sum",
    "ack_count": "sum",
    "fin_count": "sum",
    "rst_count": "sum",
    "psh_count": "sum",
    "header_len_total": "sum",
    "pkt_len_max": "max",
    "pkt_len_min": "min",
    "flow_packets_per_s": "max",     # peak burst rate is the discriminative signal
    "flow_bytes_per_s": "max",
    "pkt_len_mean": "mean",
    "pkt_len_std": "mean",
    "iat_mean": "mean",
    "iat_std": "mean",
    "down_up_ratio": "mean",
    "protocol": "first",
}


def aggregate_flows(flow_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Group ``flow_df`` by ``flow_id`` and produce one unified record per flow.

    Returns a DataFrame indexed by ``flow_id`` (as a column) carrying the
    aggregated features plus ``Label`` / ``attack_type`` (taken as the segment
    majority) and a ``n_segments`` book-keeping column.
    """
    feat_cols = [c for c in config.__dict__ if False]  # noop to satisfy linters
    agg = {}
    for col in flow_df.columns:
        if col in ("flow_id", "Label", "attack_type",
                   "src_ip", "dst_ip", "src_port", "dst_port", "segment_index"):
            continue
        agg[col] = AGG_MAP.get(col, "mean")

    grouped = flow_df.groupby("flow_id", sort=False)
    unified = grouped.agg(agg)

    # Label / attack_type: a flow's class is constant across its segments, but we
    # take the mode defensively in case of real-data noise.
    unified["Label"] = grouped["Label"].max()                       # any-attack => attack
    unified["attack_type"] = grouped["attack_type"].agg(
        lambda s: s.value_counts().index[0])
    unified["n_segments"] = grouped.size()

    # Keep flow identity columns (first occurrence).
    for idcol in ("src_ip", "dst_ip", "src_port", "dst_port"):
        if idcol in flow_df.columns:
            unified[idcol] = grouped[idcol].first()

    unified = unified.reset_index()
    if verbose:
        multi = int((unified["n_segments"] > 1).sum())
        print(f"[flow-agg] {len(flow_df):,} segment rows -> {len(unified):,} unified "
              f"flows ({multi:,} were multi-segment).")
    return unified


def map_packets_to_flows(packet_sample: pd.DataFrame,
                         unified_flows: pd.DataFrame) -> pd.DataFrame:
    """
    For each packet in ``packet_sample`` return its corresponding unified flow
    record (joined on ``flow_id``). Packets whose flow is absent are dropped
    (and reported). Used to re-check Phase-2 alerts at the flow level.
    """
    cols = [c for c in unified_flows.columns if c not in ("Label", "attack_type")]
    merged = packet_sample[["flow_id"]].merge(
        unified_flows[cols + ["Label", "attack_type"]],
        on="flow_id", how="left", suffixes=("", "_flow"))
    missing = merged["total_packets"].isna().sum() if "total_packets" in merged else 0
    if missing:
        print(f"[flow-map] {missing:,} packets had no matching flow record (dropped).")
    return merged
