"""
Supplementary analysis: flow-length vs detectability.

Addresses the report's "flow length analysis" question -- does a longer flow
(more packets / longer duration) make an attack easier to detect? We measure
the flow-level *unsupervised* detector's per-flow detection rate as a function
of total packet count and flow duration, and report the correlation.

Run with:  python -m src.analysis  [--scale demo] [--seed 42]
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

import config
from src import data_loader, evaluate, flow_aggregation
from src.preprocessing import Preprocessor, get_labels
from src.sampling import sample_dataset
from src.unsupervised import UnsupervisedDetector

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def flow_length_analysis(scale: str = "demo", seed: int = config.DEFAULT_SEED):
    from src.pipeline import SCALES
    cfg = SCALES[scale]
    config.ensure_dirs()

    _, flows, _ = data_loader.load_populations(
        seed=seed, benign_pop=cfg["benign_pop"],
        attack_pop_per_type=cfg["attack_pop_per_type"], verbose=False)
    unified = flow_aggregation.aggregate_flows(flows, verbose=False)
    sample = sample_dataset(unified, seed=seed, n_benign=cfg["n_benign"],
                            attack_min=cfg["attack_min"], attack_max=cfg["attack_max"],
                            verbose=False)

    tr, te = train_test_split(sample, test_size=0.4, random_state=seed,
                              stratify=sample["attack_type"])
    prep = Preprocessor()
    Xtr = prep.fit_transform(tr)
    Xte = prep.transform(te)
    y_te, _ = get_labels(te)

    # Use the *selective* contamination threshold (not the wide Phase-2 alert
    # budget): at a strict operating point only the most clearly-anomalous flows
    # are caught, which is exactly where a flow-length effect becomes visible.
    det = UnsupervisedDetector(seed=seed,
                               contamination=float(np.clip(tr["Label"].mean(), 0.005, 0.1)))
    det.fit(Xtr)
    pred = det.predict(Xte)

    # Restrict to attack flows; "correct" == detected.
    atk = (y_te == 1)
    df = pd.DataFrame({
        "total_packets": te["total_packets"].to_numpy()[atk],
        "flow_duration": te["flow_duration"].to_numpy()[atk],
        "attack_type": te["attack_type"].to_numpy()[atk],
        "detected": pred[atk].astype(int),
    })

    # Correlations (point-biserial == Pearson with a binary variable).
    r_pkt, p_pkt = pearsonr(np.log1p(df["total_packets"]), df["detected"])
    r_dur, p_dur = pearsonr(df["flow_duration"], df["detected"])

    # Within-attack-type correlation removes the attack-type confound
    # (packet count is correlated with which attack it is).
    within = {}
    for atk_name, g in df.groupby("attack_type"):
        if g["detected"].nunique() < 2 or len(g) < 20:
            within[atk_name] = None
            continue
        rr, pp = pearsonr(np.log1p(g["total_packets"]), g["detected"])
        within[atk_name] = {"r": float(rr), "p": float(pp), "n": int(len(g))}

    # Detection rate by packet-count quartile.
    df["pkt_bucket"] = pd.qcut(df["total_packets"], q=4, duplicates="drop")
    by_pkt = df.groupby("pkt_bucket", observed=True)["detected"].mean()

    result = {
        "scale": scale, "seed": seed, "n_attack_flows": int(atk.sum()),
        "corr_logpackets_vs_detection": {"r": float(r_pkt), "p": float(p_pkt)},
        "corr_duration_vs_detection": {"r": float(r_dur), "p": float(p_dur)},
        "within_attack_type_corr_logpackets": within,
        "detection_rate_by_packet_quartile": {str(k): float(v) for k, v in by_pkt.items()},
    }
    evaluate.save_metrics(result, f"flow_length_analysis_{scale}.json")

    # Figure: detection rate vs packet-count quartile.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(range(len(by_pkt)), by_pkt.to_numpy(), color="#4C72B0")
    ax.set_xticks(range(len(by_pkt)))
    ax.set_xticklabels([str(i) for i in by_pkt.index], rotation=15, fontsize=7)
    ax.set_ylabel("Unsupervised detection rate")
    ax.set_xlabel("Total-packets quartile (low -> high)")
    ax.set_title(f"Flow detectability vs flow length (r={r_pkt:.2f}, p={p_pkt:.1e})")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    evaluate._save(fig, "flow_length_analysis.png")

    print(f"[flow-len] corr(log packets, detected) r={r_pkt:.3f} p={p_pkt:.2e}")
    print(f"[flow-len] corr(duration, detected)     r={r_dur:.3f} p={p_dur:.2e}")
    print(f"[flow-len] detection rate by packet quartile: "
          f"{[round(float(v),3) for v in by_pkt]}")
    for k, v in within.items():
        if v:
            print(f"[flow-len]   within {k}: r={v['r']:.3f} p={v['p']:.2e} n={v['n']}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="demo")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    flow_length_analysis(scale=a.scale, seed=a.seed)
