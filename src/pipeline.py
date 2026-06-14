"""
End-to-end orchestration of the two-stage IoT IDS.

Stages
------
Phase 1  Sample an imbalanced packet dataset (Task 1.1) and preprocess it (1.2).
Phase 2  Unsupervised anomaly detection on packet features; generate + analyse
         alerts; full metric suite (Tasks 2.1, 2.2).
Phase 3  Aggregate flow segments by flow_id; draw a flow-level sample; (3.1)
         optionally run the same unsupervised method on flows and compare;
         (3.2) build flow features; (3.3) train a supervised classifier and use
         it to re-check Phase-2 alerts, reducing false positives.
Then     Comparative analysis (two-stage vs single-stage), statistical
         significance, timing/complexity, and all figures + JSON metrics.

Run with ``python run.py --scale demo``.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
from scipy.stats import chi2
from sklearn.model_selection import train_test_split

import config
from src import data_loader, evaluate, flow_aggregation
from src.preprocessing import Preprocessor, get_labels
from src.sampling import sample_dataset
from src.supervised import SupervisedIDS
from src.unsupervised import UnsupervisedDetector, select_threshold_by_fpr

# Scale presets: identical proportions/code, different magnitudes.
SCALES = {
    "smoke": dict(n_benign=4_000, attack_min=90, attack_max=130,
                  benign_pop=7_000, attack_pop_per_type=500),
    "demo": dict(n_benign=40_000, attack_min=820, attack_max=1_240,
                 benign_pop=64_000, attack_pop_per_type=2_400),
    "full": dict(n_benign=200_000, attack_min=4_000, attack_max=6_200,
                 benign_pop=320_000, attack_pop_per_type=9_000),
}


def mcnemar(y_true, pred_a, pred_b):
    """McNemar's test comparing two classifiers' errors on the same samples."""
    a_ok = (pred_a == y_true)
    b_ok = (pred_b == y_true)
    n01 = int(np.sum(a_ok & ~b_ok))      # A correct, B wrong
    n10 = int(np.sum(~a_ok & b_ok))      # A wrong, B correct
    denom = n01 + n10
    stat = ((abs(n01 - n10) - 1) ** 2 / denom) if denom > 0 else 0.0
    p = float(1 - chi2.cdf(stat, 1)) if denom > 0 else 1.0
    return {"statistic": float(stat), "p_value": p,
            "discordant_a_only": n01, "discordant_b_only": n10}


def run(scale: str = "demo", seed: int = config.DEFAULT_SEED,
        do_flow_unsup: bool = True) -> dict:
    cfg = SCALES[scale]
    config.ensure_dirs()
    rng = np.random.default_rng(seed)
    report: dict = {"scale": scale, "seed": seed}
    print(f"\n{'='*70}\nTWO-STAGE IoT IDS  |  scale={scale}  seed={seed}\n{'='*70}")

    # ===================================================================== #
    # Load populations
    # ===================================================================== #
    packets, flows, source = data_loader.load_populations(
        seed=seed, benign_pop=cfg["benign_pop"],
        attack_pop_per_type=cfg["attack_pop_per_type"])
    report["data_source"] = source

    # ===================================================================== #
    # PHASE 1 -- sampling + preprocessing (packet level)
    # ===================================================================== #
    print("\n--- PHASE 1: packet sampling + preprocessing ---")
    pkt_sample = sample_dataset(packets, seed=int(rng.integers(0, 2**31)),
                                n_benign=cfg["n_benign"],
                                attack_min=cfg["attack_min"],
                                attack_max=cfg["attack_max"])
    report["phase1"] = {
        "n_total": int(len(pkt_sample)),
        "n_attack": int(pkt_sample["Label"].sum()),
        "attack_ratio": float(pkt_sample["Label"].mean()),
    }

    train_df, test_df = train_test_split(
        pkt_sample, test_size=config.TEST_SIZE, random_state=seed,
        stratify=pkt_sample["attack_type"])

    prep = Preprocessor()
    X_tr = prep.fit_transform(train_df)
    X_te = prep.transform(test_df)
    y_tr_bin, _ = get_labels(train_df)
    y_te_bin, y_te_multi = get_labels(test_df)
    report["phase1"]["n_features"] = len(prep.feature_names)
    report["phase1"]["features"] = prep.feature_names

    # ===================================================================== #
    # PHASE 2 -- unsupervised anomaly detection on packets
    # ===================================================================== #
    print("\n--- PHASE 2: unsupervised packet-level detection ---")
    contamination = float(np.clip(train_df["Label"].mean(), 0.005, 0.1))
    t0 = time.perf_counter()
    det = UnsupervisedDetector(seed=seed, contamination=contamination).fit(X_tr)
    t_fit_p2 = time.perf_counter() - t0

    # Anomaly-IDS philosophy: cast a wide net. Flag the top ALERT_BUDGET of
    # traffic as suspicious so recall is high; Phase 3 removes the false alarms.
    ALERT_BUDGET = 0.20
    det.set_alert_budget(ALERT_BUDGET)

    t0 = time.perf_counter()
    scores_te = det.anomaly_score(X_te)
    t_pred_p2 = time.perf_counter() - t0
    phase2_pred = det.predict(X_te)

    m2 = evaluate.binary_metrics(y_te_bin, phase2_pred, scores_te)
    rates2 = evaluate.per_attack_detection_rate(y_te_multi, phase2_pred)
    report["phase2"] = {
        "contamination": contamination,
        "alert_budget": ALERT_BUDGET,
        "threshold": det.threshold_,
        "metrics": m2,
        "per_attack_detection": rates2,
        "fit_seconds": t_fit_p2,
        "predict_seconds": t_pred_p2,
    }
    print(f"[phase2] P={m2['precision']:.3f} R={m2['recall']:.3f} "
          f"F1={m2['f1']:.3f} AUC={m2.get('auc_roc', float('nan')):.3f} "
          f"FPR={m2['fpr']:.3f} FNR={m2['fnr']:.3f}")

    # Alert analysis (Task 2.2): threshold trade-off + score distribution.
    fpr_thr = select_threshold_by_fpr(scores_te, y_te_bin, config.TARGET_FPR)
    pred_fpr = (scores_te >= fpr_thr).astype(int)
    report["phase2"]["fpr_targeted_threshold"] = {
        "threshold": fpr_thr, "target_fpr": config.TARGET_FPR,
        "metrics": evaluate.binary_metrics(y_te_bin, pred_fpr, scores_te),
    }

    # Cluster structure (DoS/DDoS view) + figures.
    clusters_te = det.cluster_assignments(X_te)
    evaluate.plot_confusion(y_te_bin, phase2_pred, "Phase 2 (packet, unsupervised)",
                            "cm_phase2.png")
    evaluate.plot_score_distributions(scores_te, y_te_bin, det.threshold_,
                                      "phase2_score_dist.png")
    evaluate.plot_cluster_composition(clusters_te, y_te_multi, "phase2_clusters.png")

    # ===================================================================== #
    # PHASE 3 -- flow aggregation, sampling, (3.1) flow unsup, (3.3) supervised
    # ===================================================================== #
    print("\n--- PHASE 3: flow aggregation + supervised refinement ---")
    unified = flow_aggregation.aggregate_flows(flows)
    report["phase3"] = {"n_unified_flows": int(len(unified)),
                        "multi_segment_flows": int((unified["n_segments"] > 1).sum())}

    # Second random dataset drawn directly from flow level (same sampler).
    flow_sample = sample_dataset(unified, seed=int(rng.integers(0, 2**31)),
                                 n_benign=cfg["n_benign"],
                                 attack_min=cfg["attack_min"],
                                 attack_max=cfg["attack_max"])

    # Avoid leakage: drop flows that appear in the packet TEST set from the
    # supervised training material before splitting.
    test_flow_ids = set(test_df["flow_id"].unique())
    flow_sample = flow_sample[~flow_sample["flow_id"].isin(test_flow_ids)].reset_index(drop=True)

    f_train, f_tmp = train_test_split(flow_sample, test_size=config.TEST_SIZE,
                                      random_state=seed, stratify=flow_sample["attack_type"])
    f_val, f_test = train_test_split(f_tmp, test_size=0.5, random_state=seed,
                                     stratify=f_tmp["attack_type"])

    fprep = Preprocessor()
    Xf_tr = fprep.fit_transform(f_train)
    Xf_val = fprep.transform(f_val)
    Xf_test = fprep.transform(f_test)
    yf_tr, _ = get_labels(f_train)
    yf_val, _ = get_labels(f_val)
    yf_test, yf_test_multi = get_labels(f_test)

    # ----- Task 3.1 (optional): unsupervised on flow features ------------- #
    flow_unsup_block = None
    flow_anom_tr = flow_anom_val = flow_anom_test = None
    if do_flow_unsup:
        print("[3.1] unsupervised anomaly detection on flow features")
        f_contam = float(np.clip(f_train["Label"].mean(), 0.005, 0.1))
        fdet = UnsupervisedDetector(seed=seed, contamination=f_contam).fit(Xf_tr)
        f_scores = fdet.anomaly_score(Xf_test)
        f_pred = fdet.predict(Xf_test)
        flow_unsup_metrics = evaluate.binary_metrics(yf_test, f_pred, f_scores)
        flow_unsup_block = {
            "metrics": flow_unsup_metrics,
            "per_attack_detection": evaluate.per_attack_detection_rate(yf_test_multi, f_pred),
            "hypothesis_flow_beats_packet_auc":
                bool(flow_unsup_metrics.get("auc_roc", 0) > m2.get("auc_roc", 0)),
        }
        report["phase3"]["task3_1_flow_unsupervised"] = flow_unsup_block
        # Flow anomaly score as an engineered feature for the supervised model.
        flow_anom_tr = fdet.anomaly_score(Xf_tr).reshape(-1, 1)
        flow_anom_val = fdet.anomaly_score(Xf_val).reshape(-1, 1)
        flow_anom_test = fdet.anomaly_score(Xf_test).reshape(-1, 1)
        print(f"[3.1] flow-unsup AUC={flow_unsup_metrics.get('auc_roc', float('nan')):.3f} "
              f"vs packet-unsup AUC={m2.get('auc_roc', float('nan')):.3f}")

    # ----- Task 3.3: supervised classifier (with / without flow-anom feat) -- #
    def fit_eval_supervised(extra_tr=None, extra_val=None, extra_test=None, tag=""):
        Xtr = Xf_tr if extra_tr is None else np.hstack([Xf_tr, extra_tr])
        Xval = Xf_val if extra_val is None else np.hstack([Xf_val, extra_val])
        Xtst = Xf_test if extra_test is None else np.hstack([Xf_test, extra_test])
        t0 = time.perf_counter()
        clf = SupervisedIDS(seed=seed).fit(Xtr, yf_tr, Xval, yf_val)
        t_fit = time.perf_counter() - t0
        thr = clf.tune_threshold(Xval, yf_val)
        t0 = time.perf_counter()
        proba = clf.predict_proba(Xtst)
        t_pred = time.perf_counter() - t0
        pred = (proba >= thr).astype(int)
        block = {
            "backend": clf.backend, "threshold": thr,
            "metrics": evaluate.binary_metrics(yf_test, pred, proba),
            "per_attack_detection": evaluate.per_attack_detection_rate(yf_test_multi, pred),
            "fit_seconds": t_fit, "predict_seconds": t_pred,
        }
        print(f"[3.3{tag}] sup F1={block['metrics']['f1']:.3f} "
              f"AUC={block['metrics']['auc_roc']:.3f} "
              f"FPR={block['metrics']['fpr']:.3f}")
        return clf, thr, block

    clf_base, thr_base, block_base = fit_eval_supervised(tag=" base")
    report["phase3"]["task3_3_supervised_without_flow_anomaly"] = block_base
    sup_curve = (yf_test, clf_base.predict_proba(Xf_test))

    clf_main, thr_main, clf_for_cascade = clf_base, thr_base, clf_base
    if do_flow_unsup:
        clf_aug, thr_aug, block_aug = fit_eval_supervised(
            flow_anom_tr, flow_anom_val, flow_anom_test, tag=" +anom")
        report["phase3"]["task3_3_supervised_with_flow_anomaly"] = block_aug
        # Use the augmented model for the cascade if it validated better.
        if block_aug["metrics"]["f1"] >= block_base["metrics"]["f1"]:
            clf_for_cascade, thr_main = clf_aug, thr_aug
            report["phase3"]["cascade_model"] = "with_flow_anomaly"
        else:
            report["phase3"]["cascade_model"] = "without_flow_anomaly"
    else:
        report["phase3"]["cascade_model"] = "without_flow_anomaly"

    # feature importances (top 12)
    try:
        imp = clf_base.feature_importance(fprep.feature_names)
        report["phase3"]["top_flow_features"] = dict(list(imp.items())[:12])
    except Exception as e:                                   # pragma: no cover
        report["phase3"]["top_flow_features_error"] = str(e)

    # ===================================================================== #
    # TWO-STAGE CASCADE on the packet TEST set
    # ===================================================================== #
    print("\n--- TWO-STAGE CASCADE: re-check Phase-2 alerts at flow level ---")
    # Look up each packet-test row's unified flow record, transform identically.
    mapped = test_df[["flow_id"]].merge(unified, on="flow_id", how="left")
    have_flow = mapped["total_packets"].notna().to_numpy()
    Xcasc = fprep.transform(mapped.ffill().bfill())
    if do_flow_unsup and report["phase3"]["cascade_model"] == "with_flow_anomaly":
        casc_anom = fdet.anomaly_score(Xcasc).reshape(-1, 1)
        flow_proba = clf_for_cascade.predict_proba(np.hstack([Xcasc, casc_anom]))
    else:
        flow_proba = clf_for_cascade.predict_proba(Xcasc)
    phase3_confirm = (flow_proba >= thr_main).astype(int)
    phase3_confirm[~have_flow] = phase2_pred[~have_flow]    # no flow -> trust Phase 2

    combined_pred = (phase2_pred & phase3_confirm).astype(int)

    # Single-stage baselines for comparison.
    supervised_only = phase3_confirm.copy()                 # flow model on all packets

    m_combined = evaluate.binary_metrics(y_te_bin, combined_pred, scores_te)
    m_suponly = evaluate.binary_metrics(y_te_bin, supervised_only, flow_proba)
    rates_combined = evaluate.per_attack_detection_rate(y_te_multi, combined_pred)

    fp_p2 = m2["fp"]
    fp_comb = m_combined["fp"]
    fp_reduction = (fp_p2 - fp_comb) / fp_p2 if fp_p2 else 0.0

    report["cascade"] = {
        "phase2_only": m2,
        "supervised_only": m_suponly,
        "two_stage_combined": m_combined,
        "per_attack_detection_combined": rates_combined,
        "false_positive_reduction_pct": 100.0 * fp_reduction,
        "fp_phase2": fp_p2, "fp_combined": fp_comb,
        "system_accuracy": m_combined["accuracy"],
        "mcnemar_combined_vs_phase2": mcnemar(y_te_bin, phase2_pred, combined_pred),
        "timing": {
            "phase2_fit_s": t_fit_p2, "phase2_predict_s": t_pred_p2,
            "phase3_fit_s": block_base["fit_seconds"],
            "phase3_predict_s": block_base["predict_seconds"],
            # In production Phase 3 only scores the packets Phase 2 flagged.
            "n_packets_scored_phase2": int(len(y_te_bin)),
            "n_packets_flagged_for_phase3": int(phase2_pred.sum()),
            "phase3_workload_fraction": float(phase2_pred.mean()),
        },
    }
    print(f"[cascade] Phase2-only: P={m2['precision']:.3f} R={m2['recall']:.3f} "
          f"FP={fp_p2}")
    print(f"[cascade] Two-stage : P={m_combined['precision']:.3f} "
          f"R={m_combined['recall']:.3f} FP={fp_comb}")
    print(f"[cascade] FALSE-POSITIVE REDUCTION: {100*fp_reduction:.1f}%   "
          f"system accuracy={m_combined['accuracy']:.4f}")

    # ===================================================================== #
    # Comparative figures
    # ===================================================================== #
    evaluate.plot_confusion(y_te_bin, combined_pred, "Two-stage (combined)",
                            "cm_combined.png")
    evaluate.plot_roc({
        "Phase 2 packet-unsup": (y_te_bin, scores_te),
        "Flow supervised": sup_curve,
    }, "roc_compare.png")
    evaluate.plot_pr({
        "Phase 2 packet-unsup": (y_te_bin, scores_te),
        "Flow supervised": sup_curve,
    }, "pr_compare.png")
    evaluate.plot_per_attack(rates2, rates_combined, "per_attack_compare.png")

    evaluate.save_metrics(report, f"results_{scale}.json")
    print(f"\n{'='*70}\nDONE. Metrics -> results/metrics/results_{scale}.json, "
          f"figures -> results/figures/\n{'='*70}")
    return report
