"""
Schema-faithful synthetic generator for the CIC IoT-DIAD 2024 dataset.

This module is a *stand-in* for the real download. It produces two coherent
tables that mirror the structure documented in the project brief:

  * a **packet-level** table  (one row per packet), and
  * a **flow-level**   table  (one row per 2-minute flow segment),

linked by a ``flow_id`` of the form ``src_ip-dst_ip-src_port-dst_port``. Flows
whose duration exceeds 120 s are split into successive 2-minute segments that
share the same ``flow_id`` -- exactly the behaviour the brief describes -- so
the Phase-3 group-by-flow-id aggregation has something real to do.

Design goals (so the downstream ML story is meaningful, not trivial):
  * Each attack type has a distinct but *overlapping* signature, so detection is
    non-trivial and per-attack performance varies.
  * Flow-level aggregate features (packets/s, duration, totals) encode the class
    signal more cleanly than noisy per-packet features. This is what lets us
    test the brief's hypothesis that flow features help.
  * Benign and attack distributions overlap enough that the unsupervised stage
    makes mistakes, leaving room for the supervised stage to reduce false
    positives.

When the real CSVs are present under ``DATA_DIR`` the loader uses them instead;
this generator is only a fallback. The two tables it returns use the same
column names the rest of the pipeline expects for real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# --------------------------------------------------------------------------- #
# Column schemas (kept explicit so real-data adapters can map onto them)
# --------------------------------------------------------------------------- #
PACKET_ID_COLS = ["flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "timestamp"]
PACKET_FEATURES = [
    "protocol", "packet_length", "header_length", "payload_length", "ttl",
    "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg",
    "iat_ms", "inst_rate", "window_size",
]

FLOW_ID_COLS = ["flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "segment_index"]
FLOW_FEATURES = [
    "protocol", "flow_duration", "total_packets", "total_bytes",
    "fwd_packets", "bwd_packets", "pkt_len_mean", "pkt_len_std",
    "pkt_len_min", "pkt_len_max", "iat_mean", "iat_std",
    "flow_bytes_per_s", "flow_packets_per_s",
    "syn_count", "ack_count", "fin_count", "rst_count", "psh_count",
    "down_up_ratio", "header_len_total",
]
LABEL_COLS = ["Label", "attack_type"]


# --------------------------------------------------------------------------- #
# Per-class latent parameters
# --------------------------------------------------------------------------- #
# Each entry describes how a flow of that class behaves. Values are deliberately
# rooted in network intuition (see README / report for the rationale).
_CLASS_PARAMS = {
    config.BENIGN_LABEL: dict(
        # IoT benign traffic is comparatively regular/periodic, so the benign
        # manifold is fairly tight -- this is exactly what makes unsupervised
        # anomaly detection viable on IoT networks.
        proto_choices=[6, 17, 1], proto_p=[0.72, 0.24, 0.04],   # TCP/UDP/ICMP
        dport_pool=[80, 443, 53, 123, 22, 8080, 5353, 1900],
        dur_mean=18.0, dur_long_frac=0.06,
        npkt_mean=3.0, npkt_cap=40,
        plen_mean=470, plen_std=150,
        iat_mean=40.0, iat_std=33.0,
        syn_p=0.12, ack_p=0.6, fin_p=0.08, rst_p=0.02, psh_p=0.38, urg_p=0.005,
        rate_mean=6.0, ttl_mean=62, src_distinct=0.4,
    ),
    "DDoS-HTTP_Flood": dict(
        proto_choices=[6], proto_p=[1.0],
        dport_pool=[80, 8080],
        dur_mean=9.0, dur_long_frac=0.02,
        npkt_mean=22.0, npkt_cap=60,
        plen_mean=140, plen_std=45,
        iat_mean=2.0, iat_std=2.5,
        syn_p=0.75, ack_p=0.8, fin_p=0.05, rst_p=0.05, psh_p=0.6, urg_p=0.0,
        rate_mean=380.0, ttl_mean=52, src_distinct=0.95,   # many distinct sources
    ),
    "DoS-HTTP_Flood": dict(
        proto_choices=[6], proto_p=[1.0],
        dport_pool=[80, 8080],
        dur_mean=11.0, dur_long_frac=0.03,
        npkt_mean=18.0, npkt_cap=60,
        plen_mean=155, plen_std=50,
        iat_mean=3.0, iat_std=3.0,
        syn_p=0.7, ack_p=0.78, fin_p=0.06, rst_p=0.05, psh_p=0.58, urg_p=0.0,
        rate_mean=240.0, ttl_mean=54, src_distinct=0.15,   # few sources
    ),
    "DNS_Spoofing": dict(
        proto_choices=[17], proto_p=[1.0],
        dport_pool=[53, 5353],
        dur_mean=6.0, dur_long_frac=0.01,
        npkt_mean=3.5, npkt_cap=30,
        plen_mean=95, plen_std=28,
        iat_mean=12.0, iat_std=18.0,
        syn_p=0.0, ack_p=0.0, fin_p=0.0, rst_p=0.0, psh_p=0.05, urg_p=0.0,
        rate_mean=30.0, ttl_mean=58, src_distinct=0.5,
    ),
    "XSS": dict(
        proto_choices=[6], proto_p=[1.0],
        dport_pool=[80, 443],
        dur_mean=42.0, dur_long_frac=0.18,                  # longer sessions
        npkt_mean=7.0, npkt_cap=50,
        plen_mean=900, plen_std=420,                        # large script payloads
        iat_mean=60.0, iat_std=80.0,
        syn_p=0.2, ack_p=0.7, fin_p=0.15, rst_p=0.04, psh_p=0.75, urg_p=0.02,
        rate_mean=6.0, ttl_mean=60, src_distinct=0.35,
    ),
    "Brute_Force": dict(
        proto_choices=[6], proto_p=[1.0],
        dport_pool=[22, 21, 3389, 23],
        dur_mean=55.0, dur_long_frac=0.20,                  # many repeated attempts
        npkt_mean=11.0, npkt_cap=50,
        plen_mean=110, plen_std=35,
        iat_mean=25.0, iat_std=30.0,
        syn_p=0.6, ack_p=0.5, fin_p=0.25, rst_p=0.4, psh_p=0.3, urg_p=0.0,
        rate_mean=20.0, ttl_mean=56, src_distinct=0.3,
    ),
}


def _rand_ips(rng: np.random.Generator, n: int, distinct: float) -> np.ndarray:
    """Generate `n` IPv4-like strings. `distinct` ~1 => many unique sources."""
    pool_size = max(2, int(n * distinct))
    octets = rng.integers(1, 254, size=(pool_size, 4))
    ip_pool = np.array(["{}.{}.{}.{}".format(*o) for o in octets])
    idx = rng.integers(0, pool_size, size=n)
    return ip_pool[idx]


def _gen_class_flows(class_name: str, n_flows: int, rng: np.random.Generator):
    """Return a dict of per-flow latent arrays for one class."""
    p = _CLASS_PARAMS[class_name]

    proto = rng.choice(p["proto_choices"], size=n_flows, p=p["proto_p"])
    src_ip = _rand_ips(rng, n_flows, p["src_distinct"])
    dst_ip = _rand_ips(rng, n_flows, 0.2)
    src_port = rng.integers(1024, 65535, size=n_flows)
    dst_port = rng.choice(p["dport_pool"], size=n_flows)

    # Duration: exponential body + a long tail (forces multi-segment flows).
    dur = rng.exponential(p["dur_mean"], size=n_flows)
    long_mask = rng.random(n_flows) < p["dur_long_frac"]
    dur[long_mask] = rng.uniform(140, 600, size=long_mask.sum())
    dur = np.clip(dur, 0.2, 600.0)

    # Packet count per flow (Poisson, capped).
    npkt = rng.poisson(p["npkt_mean"], size=n_flows) + 1
    npkt = np.clip(npkt, 1, p["npkt_cap"]).astype(int)

    plen_base = rng.normal(p["plen_mean"], p["plen_std"], size=n_flows)
    plen_base = np.clip(plen_base, 40, 1500)
    iat_base = np.abs(rng.normal(p["iat_mean"], p["iat_std"], size=n_flows)) + 0.1
    rate = np.abs(rng.normal(p["rate_mean"], p["rate_mean"] * 0.3, size=n_flows)) + 0.5
    ttl = np.clip(rng.normal(p["ttl_mean"], 6, size=n_flows), 16, 128)

    flow_id = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
        src_ip, "-"), dst_ip), "-"),
        src_port.astype(str)), np.char.add("-", dst_port.astype(str)))

    return dict(
        flow_id=flow_id, src_ip=src_ip, dst_ip=dst_ip,
        src_port=src_port, dst_port=dst_port, proto=proto,
        dur=dur, npkt=npkt, plen_base=plen_base, iat_base=iat_base,
        rate=rate, ttl=ttl, params=p, class_name=class_name,
    )


def _build_flow_rows(F: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Aggregate latent flow params into 2-minute flow-segment rows."""
    p = F["params"]
    n = len(F["flow_id"])
    seg_count = np.ceil(F["dur"] / 120.0).astype(int)
    seg_count = np.clip(seg_count, 1, 5)

    rows = []
    # Expand each flow into its segments (vectorised per-flow attributes via repeat).
    rep = np.repeat(np.arange(n), seg_count)
    seg_index = np.concatenate([np.arange(c) for c in seg_count])

    dur_full = F["dur"][rep]
    seg_dur = np.minimum(120.0, dur_full - seg_index * 120.0)
    seg_dur = np.clip(seg_dur, 0.2, 120.0)
    frac = seg_dur / dur_full                       # share of the flow in this segment

    npkt_full = F["npkt"][rep].astype(float)
    seg_pkts = np.maximum(1, np.round(npkt_full * frac)).astype(int)
    plen_mean = np.clip(F["plen_base"][rep] + rng.normal(0, 18, size=len(rep)), 40, 1500)
    plen_std = np.abs(rng.normal(p["plen_std"] * 0.5, p["plen_std"] * 0.15, size=len(rep))) + 1
    iat_mean = F["iat_base"][rep] + np.abs(rng.normal(0, 2, size=len(rep)))
    iat_std = np.abs(rng.normal(p["iat_std"] * 0.6, p["iat_std"] * 0.2, size=len(rep))) + 0.1

    total_bytes = seg_pkts * plen_mean
    flow_pkts_per_s = seg_pkts / seg_dur
    flow_bytes_per_s = total_bytes / seg_dur
    fwd = np.round(seg_pkts * rng.uniform(0.4, 0.7, size=len(rep))).astype(int)
    bwd = np.maximum(0, seg_pkts - fwd)
    down_up = (bwd + 1) / (fwd + 1)

    syn = rng.binomial(seg_pkts, p["syn_p"])
    ack = rng.binomial(seg_pkts, p["ack_p"])
    fin = rng.binomial(seg_pkts, p["fin_p"])
    rst = rng.binomial(seg_pkts, p["rst_p"])
    psh = rng.binomial(seg_pkts, p["psh_p"])
    header_total = seg_pkts * rng.normal(32, 4, size=len(rep))

    df = pd.DataFrame({
        "flow_id": F["flow_id"][rep],
        "src_ip": F["src_ip"][rep], "dst_ip": F["dst_ip"][rep],
        "src_port": F["src_port"][rep], "dst_port": F["dst_port"][rep],
        "segment_index": seg_index,
        "protocol": F["proto"][rep],
        "flow_duration": seg_dur,
        "total_packets": seg_pkts,
        "total_bytes": total_bytes,
        "fwd_packets": fwd, "bwd_packets": bwd,
        "pkt_len_mean": plen_mean, "pkt_len_std": plen_std,
        "pkt_len_min": np.clip(plen_mean - 2 * plen_std, 20, None),
        "pkt_len_max": plen_mean + 2 * plen_std,
        "iat_mean": iat_mean, "iat_std": iat_std,
        "flow_bytes_per_s": flow_bytes_per_s,
        "flow_packets_per_s": flow_pkts_per_s,
        "syn_count": syn, "ack_count": ack, "fin_count": fin,
        "rst_count": rst, "psh_count": psh,
        "down_up_ratio": down_up,
        "header_len_total": header_total,
        "Label": np.where(F["class_name"] == config.BENIGN_LABEL, 0, 1),
        "attack_type": F["class_name"],
    })
    return df


def _build_packet_rows(F: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Expand each flow into individual (noisy) packet rows."""
    p = F["params"]
    n = len(F["flow_id"])
    npkt = F["npkt"]
    total = int(npkt.sum())
    rep = np.repeat(np.arange(n), npkt)

    plen = np.clip(F["plen_base"][rep] + rng.normal(0, p["plen_std"] * 0.6, size=total), 40, 1500)
    header = np.clip(rng.normal(32, 5, size=total), 20, 60)
    payload = np.clip(plen - header, 0, None)
    ttl = np.clip(F["ttl"][rep] + rng.normal(0, 3, size=total), 8, 128)
    iat = np.abs(F["iat_base"][rep] + rng.normal(0, p["iat_std"] * 0.7, size=total)) + 0.05
    inst_rate = np.abs(F["rate"][rep] + rng.normal(0, F["rate"][rep] * 0.25, size=total)) + 0.1
    window = np.clip(rng.normal(8192, 3000, size=total), 0, 65535)

    df = pd.DataFrame({
        "flow_id": F["flow_id"][rep],
        "src_ip": F["src_ip"][rep], "dst_ip": F["dst_ip"][rep],
        "src_port": F["src_port"][rep], "dst_port": F["dst_port"][rep],
        "timestamp": rng.uniform(0, 1e6, size=total),
        "protocol": F["proto"][rep],
        "packet_length": plen, "header_length": header, "payload_length": payload,
        "ttl": ttl,
        "flag_syn": rng.binomial(1, p["syn_p"], size=total),
        "flag_ack": rng.binomial(1, p["ack_p"], size=total),
        "flag_fin": rng.binomial(1, p["fin_p"], size=total),
        "flag_rst": rng.binomial(1, p["rst_p"], size=total),
        "flag_psh": rng.binomial(1, p["psh_p"], size=total),
        "flag_urg": rng.binomial(1, p["urg_p"], size=total),
        "iat_ms": iat, "inst_rate": inst_rate, "window_size": window,
        "Label": np.where(F["class_name"] == config.BENIGN_LABEL, 0, 1),
        "attack_type": F["class_name"],
    })
    return df


def generate(seed: int = config.DEFAULT_SEED,
             benign_pop: int | None = None,
             attack_pop_per_type: int | None = None):
    """
    Generate coherent packet-level and flow-level populations.

    Returns
    -------
    (packets_df, flows_df) : tuple[pd.DataFrame, pd.DataFrame]
    """
    rng = np.random.default_rng(seed)
    benign_pop = benign_pop or config.SYNTH_BENIGN_POP
    attack_pop_per_type = attack_pop_per_type or config.SYNTH_ATTACK_POP_PER_TYPE

    packet_frames, flow_frames = [], []
    for class_name in config.ALL_CLASSES:
        n_flows = benign_pop if class_name == config.BENIGN_LABEL else attack_pop_per_type
        F = _gen_class_flows(class_name, n_flows, rng)
        packet_frames.append(_build_packet_rows(F, rng))
        flow_frames.append(_build_flow_rows(F, rng))

    packets = pd.concat(packet_frames, ignore_index=True)
    flows = pd.concat(flow_frames, ignore_index=True)

    # Shuffle so rows are not class-ordered.
    packets = packets.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    flows = flows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return packets, flows
