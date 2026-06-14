"""
GROUP: CASCADE + INTEGRATION + EVALUATION + REPORT
==================================================
You own the two-stage glue, the full evaluation suite, and the report.

You build EVERYTHING against MOCKS first -- you do NOT wait for the other groups:
    python -m team_kit.group_cascade_starter            # uses mocks
At integration, the other groups drop their .joblib bundles into
team_kit/artifacts/ and the SAME script uses them automatically:
    python -m team_kit.group_cascade_starter            # uses real models

You never preprocess model inputs yourself -- each bundle carries its own fitted
preprocessor. You only wire stages together and evaluate.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import train_test_split

from team_kit import contract
from team_kit.mocks import MockClassifier, MockDetector, MockTransformer


def _load_or_mock():
    """Use real bundles if present, else mocks. Lets you start on day 1."""
    if os.path.exists(contract.UNSUP_BUNDLE):
        ub = contract.load_bundle(contract.UNSUP_BUNDLE)
        unsup_prep, detector = ub.preprocessor, ub.model
        print("[cascade] using REAL unsupervised bundle")
    else:
        unsup_prep, detector = MockTransformer(), MockDetector()
        print("[cascade] using MOCK unsupervised model")
    if os.path.exists(contract.SUP_BUNDLE):
        sb = contract.load_bundle(contract.SUP_BUNDLE)
        sup_prep, clf, sup_thr = sb.preprocessor, sb.model, sb.extra.get("threshold", 0.5)
        print("[cascade] using REAL supervised bundle")
    else:
        sup_prep, clf, sup_thr = MockTransformer(), MockClassifier(), 0.5
        print("[cascade] using MOCK supervised model")
    return unsup_prep, detector, sup_prep, clf, sup_thr


def mcnemar(y, a, b):
    n01 = int(np.sum((a == y) & (b != y)))
    n10 = int(np.sum((a != y) & (b == y)))
    d = n01 + n10
    stat = ((abs(n01 - n10) - 1) ** 2 / d) if d else 0.0
    return {"statistic": stat, "p_value": float(1 - chi2.cdf(stat, 1)) if d else 1.0,
            "fixed_by_two_stage": n10, "broken_by_two_stage": n01}


def main(seed: int = 42, target_cascade_recall: float = 0.95):
    unsup_prep, detector, sup_prep, clf, sup_thr = _load_or_mock()

    pkt, unified = contract.get_packet_and_flows(seed)
    tr, te = train_test_split(pkt, test_size=0.3, random_state=seed,
                              stratify=pkt[contract.ATTACK_TYPE_COL])
    y = te[contract.LABEL_COL].to_numpy()
    y_multi = te[contract.ATTACK_TYPE_COL].to_numpy()

    # --- STAGE 1: packet-level alerts (wide net) ---
    detector.fit(unsup_prep.fit_transform(tr))     # mock/real both expose fit/transform
    phase2 = detector.predict(unsup_prep.transform(te))

    # --- STAGE 2: re-check each flagged packet's flow ---
    mapped = te[[contract.FLOW_ID_COL]].merge(unified, on=contract.FLOW_ID_COL,
                                              how="left", indicator=True)
    have_flow = (mapped["_merge"] == "both").to_numpy()
    mapped = mapped.drop(columns="_merge")
    flow_proba = clf.predict_proba(sup_prep.transform(mapped.ffill().bfill()))

    # Confirmation threshold. Baseline: use the threshold the supervised group
    # shipped (tuned for F1). ENHANCEMENT (recommended, see src/pipeline.py): make
    # it recall-preserving -- set thr so >= target_cascade_recall of *known attack*
    # flows are confirmed, using a labelled flow holdout:
    #     atk_p = clf.predict_proba(sup_prep.transform(holdout[holdout.Label==1]))
    #     thr   = min(sup_thr, np.quantile(atk_p, 1 - target_cascade_recall))
    thr = sup_thr
    confirm = (flow_proba >= thr).astype(int)
    confirm[~have_flow] = phase2[~have_flow]       # no flow -> trust stage 1
    combined = (phase2 & confirm).astype(int)

    # --- EVALUATE ---
    m2 = contract.binary_metrics(y, phase2)
    mc = contract.binary_metrics(y, combined)
    fp_red = 100 * (m2["fp"] - mc["fp"]) / m2["fp"] if m2["fp"] else 0.0
    print(f"\n[Phase2-only ] P={m2['precision']:.3f} R={m2['recall']:.3f} FP={m2['fp']}")
    print(f"[Two-stage   ] P={mc['precision']:.3f} R={mc['recall']:.3f} FP={mc['fp']}")
    print(f"[FP reduction] {fp_red:.1f}%   accuracy={mc['accuracy']:.4f}")
    print(f"[McNemar     ] {mcnemar(y, phase2, combined)}")
    print("[per-attack  ]",
          {k: round(v["detection_rate"], 3)
           for k, v in contract.per_attack_detection_rate(y_multi, combined).items()})


if __name__ == "__main__":
    main()
