# Multi-Stage IoT Network Intrusion Detection
## A Two-Stage Unsupervised + Supervised IDS on CIC IoT-DIAD 2024

**ECE 597 Capstone — Summer 2026**

---

## 1. Introduction and Methodology

### 1.1 Problem statement

IoT networks generate enormous volumes of packet traffic, the overwhelming
majority of which is benign. A practical intrusion-detection system (IDS) must
(a) catch attacks it has never been explicitly told about, and (b) avoid
drowning operators in false alarms. These two goals pull in opposite
directions: a sensitive detector raises many false positives, while a
conservative one misses attacks.

We address this with a **two-stage hybrid IDS** that separates the two concerns:

* **Stage 1 — Anomaly-based IDS (unsupervised, packet level).** Learns the
  benign traffic manifold *without labels* and flags anything that deviates.
  It is tuned for **high recall** and deliberately tolerates false positives —
  it casts a wide net.
* **Stage 2 — Signature-based IDS (supervised, flow level).** A trained
  classifier **re-checks every Stage-1 alert** using richer, aggregated
  flow-level features, and discards the false positives.

The composition keeps Stage 1's detection rate while sharply cutting its false
positives, and — because Stage 2 only ever sees the ~20 % of traffic Stage 1
flagged — it does so at a fraction of the cost of classifying every packet.

### 1.2 Dataset

We target the **CIC IoT-DIAD 2024** dataset, which provides both packet-level
and flow-level captures across several attack scenarios. Per the brief we use
five attack categories — **DDoS-HTTP Flood, DoS-HTTP Flood, DNS Spoofing, XSS,
and Brute Force** — choosing the HTTP-Flood variants for DoS/DDoS because the
TCP-Flood flow data is absent from this release.

> **Reproducibility note on the data.** The real CIC CSVs must be downloaded
> from the CIC server. The development environment's network policy blocked that
> host, so the codebase includes a **schema-faithful synthetic generator**
> (`src/data_synth.py`) that reproduces the dataset's structure — two tables
> linked by `flow_id = src_ip-dst_ip-src_port-dst_port`, with flows longer than
> 120 s split into 2-minute segments sharing a `flow_id`. The loader uses the
> real files automatically when present; **no downstream code changes**. All
> numbers in this report are from the synthetic stand-in at the full spec scale
> (200,000 benign rows); they exercise the entire methodology and should be read
> as a demonstration of the system, not as published CIC benchmarks.

### 1.3 Methodology overview

```
            packet CSV                         flow CSV
                │                                  │
       Task 1.1 sample (200k benign, 2–3% attack)  │ group by flow_id
                │                                  │ (sum/max/mean aggregation)
       Task 1.2 preprocess                  unified flow records
                │                                  │
   ┌────────────▼─────────────┐         Task 1.1 sample (200k benign)
   │ PHASE 2 (unsupervised)   │                    │
   │ AE + IsolationForest +   │         ┌──────────▼──────────────┐
   │ KMeans  →  ALERTS        │         │ PHASE 3 (supervised)    │
   └────────────┬─────────────┘         │ XGBoost on flow feats   │
                │  flagged packets       │ (+ optional flow-anomaly│
                └──────────►  look up flow record  ►  RE-CHECK ◄───┘
                                         │
                              two-stage decision = Stage1 ∧ Stage2
```

---

## 2. Implementation Details

### 2.1 Phase 1 — sampling and preprocessing

**Task 1.1 — Random sampler (`src/sampling.py`).** `sample_dataset` draws
200,000 benign rows and a random `4,000–6,200` total attack rows (≈ 2–3 %).
The attack budget is spread **roughly uniformly across the five types**
(each centred on `total/5` with a ±15 % random jitter, every type guaranteed
≥ 1 row), so each run produces a *different attack composition* while keeping
an even split and preserving the imbalance ratio.
The function performs integrity checks (required columns, sufficient pool sizes,
exact benign count, attack ceiling) and is reused unchanged for both the
packet-level and flow-level samples. A representative draw:

```
total=205,754  benign=200,000  attack=5,754 (2.80%)
per-attack: DDoS=1128  DoS=1065  DNS=1289  XSS=1232  BruteForce=1040
```

**Task 1.2 — Preprocessing (`src/preprocessing.py`).** A fit-once / apply-many
transformer with the following deliberately transparent steps, each justified:

| Step | Choice | Why |
|---|---|---|
| Drop identifiers | remove `flow_id`, IPs, raw ports, `timestamp`, `segment_index` | prevent the model memorising hosts (label leakage) instead of learning behaviour |
| Coerce + impute | to numeric, then **median** imputation | rate features are NaN on 1-packet flows; median is robust to skew |
| Replace ±∞ | → NaN → imputed | bytes/s is infinite on 0-duration flows |
| Robust clip | to 1st/99th percentile | volumetric floods produce legitimate but enormous values that would otherwise dominate scaling |
| Standard-scale | zero mean, unit variance | required by distance-based (k-means) and gradient-based (autoencoder) methods |

Identifier columns are excluded from the model matrix but kept on the side so
the Phase-3 stage can still join packets to their flows. The packet sample
yields **16 numeric features**.

### 2.2 Phase 2 — unsupervised anomaly detection (`src/unsupervised.py`)

We combine **three complementary unsupervised signals**, each z-scored and
averaged into one anomaly score:

1. **Autoencoder reconstruction error.** An MLP autoencoder
   (`16→32→12→32→16`, ReLU, Adam) trained to reconstruct its input. Because
   benign IoT traffic is comparatively regular it dominates the learned
   manifold, so attacks reconstruct poorly.
2. **Isolation Forest** (200 trees) path-length score — isolates globally rare
   points.
3. **K-means distance-to-centroid** on the AE-augmented feature space
   (features + reconstruction error). We deliberately **over-cluster** (k = 8)
   so cluster membership exposes structure (Section 4.5).

No labels are used in training. **Threshold (Task 2.2).** An anomaly-based IDS
should be *sensitive*, so the operational threshold flags the top
**`ALERT_BUDGET = 20 %`** of scores — a wide net that maximises recall and
hands the false-positive problem to Phase 3. We additionally report a
label-aware **FPR-targeted** threshold (≤ 5 % benign FPR) purely to illustrate
the precision/recall trade-off.

### 2.3 Phase 3 — flow aggregation + supervised refinement

**Flow segmentation (`src/flow_aggregation.py`).** Flows are grouped by
`flow_id` and collapsed to one record using a **per-feature aggregation policy**
that respects each feature's meaning:

* **sum** — additive quantities (packet/byte/flag counts, duration);
* **max** — peak quantities (peak packets/s and bytes/s, max packet length);
* **mean** — rates / ratios / averages.

In the full run, **24,905 of 365,000 flows (6.8 %) spanned multiple segments**,
so the aggregation is doing real work rather than a no-op `groupby`.

**Supervised model (`src/supervised.py`).** Gradient-boosted trees (**XGBoost**,
400 trees, depth 6, `scale_pos_weight` set to the negative/positive ratio to
counter imbalance; a `HistGradientBoosting` fallback exists if XGBoost is
unavailable). Trees are a strong, fast, well-understood baseline for tabular
flow features and are robust to mixed scales. We use a **train/validation/test
split** (70/15/15), tune the decision threshold to maximise validation F1, and
remove any flows present in the packet test set from the supervised training
material to prevent leakage in the cascade evaluation.

**Task 3.1 (optional) — flow-level unsupervised.** The same AE+IF+KMeans
ensemble is run on flow features, both to test the "flow features help"
hypothesis (Section 4.2) and to produce a **flow-anomaly score** that is offered
to the supervised model as an extra engineered feature (with/without ablation,
Section 4.4).

---

## 3. Results and Analysis

All figures are in `results/figures/`; all numbers in
`results/metrics/results_full.json`.

### 3.1 Phase 2 — packet-level unsupervised detection

| Metric | Value |
|---|---|
| Precision | 0.123 |
| Recall | **0.886** |
| F1 | 0.216 |
| AUC-ROC | **0.917** |
| FPR | 0.182 |
| FNR | 0.114 |
| Accuracy | 0.820 |
| Confusion (TP/FP/TN/FN) | 1529 / 10908 / 49093 / 197 |

The low precision is **by design**: at a 20 % alert budget the detector raises
10,908 false positives to achieve 89 % recall and AUC 0.917. This is the wide
net the second stage will clean up. (Figures: `cm_phase2.svg`,
`phase2_score_dist.svg`.)

**Per-attack detection rate (packet level):**

| Attack | Detection rate |
|---|---|
| DoS-HTTP Flood | 0.997 |
| DDoS-HTTP Flood | 0.988 |
| DNS Spoofing | 0.969 |
| Brute Force | 0.923 |
| XSS | **0.578** |

Volumetric and protocol-anomalous attacks (floods, DNS spoofing, brute force)
are highly detectable from individual packets; **XSS is the hardest** at the
packet level because its malicious character lives in the *session/payload*
pattern, not in any single packet (revisited in Section 4.3).

### 3.2 Phase 3 — supervised flow classifier

| Model | F1 | AUC | FPR |
|---|---|---|---|
| Flow supervised (base) | **0.9944** | 1.000 | 0.000 |
| Flow supervised (+ flow-anomaly feature) | 0.9945 | 1.000 | 0.000 |

The flow classifier is near-perfect on these features. The most important flow
features (XGBoost gain) are interpretable: `pkt_len_max` (0.60), `pkt_len_std`
(0.22), `psh_count`, `iat_std`, `total_packets`.

### 3.3 Two-stage system vs single-stage baselines

| System | Precision | Recall | False Positives | Accuracy |
|---|---|---|---|---|
| Stage 1 only (packet, unsupervised) | 0.123 | 0.886 | **10,908** | 0.820 |
| Single-stage supervised (flow, all packets) | 0.976 | **0.999** | 42 | 0.999 |
| **Two-stage (Stage 1 ∧ Stage 2)** | **0.993** | 0.885 | **10** | 0.997 |

**False-positive reduction: 10,908 → 10 = 99.91 %**, with recall preserved
(0.886 → 0.885 — only true positives Stage 1 happened to miss are lost, none are
introduced). Figures: `cm_combined.svg`, `roc_compare.svg`, `pr_compare.svg`,
`per_attack_compare.svg`.

### 3.4 Statistical significance

**McNemar's test**, two-stage vs Stage-1-only on the 61,727-packet test set:
χ² = 10,893, **p ≈ 0**. The discordances are almost entirely one-directional:
the two-stage system **corrects 10,898 of Stage 1's errors while introducing
only 1**. The improvement is overwhelmingly significant, not noise.

### 3.5 Computational overhead and time complexity

| Stage | Fit time | Predict time | Notes |
|---|---|---|---|
| Phase 2 (unsupervised) | 10.51 s | 0.62 s | AE training dominates |
| Phase 3 (supervised) | 1.39 s | 0.03 s | trees train fast |

**Workload of the cascade:** Phase 3 only scores the **20.1 %** of packets that
Phase 2 flagged (12,437 of 61,727), so the expensive supervised stage runs on a
fifth of the traffic. Asymptotically, Phase-2 inference is `O(N·(d + T))` (AE
forward pass + `T` isolation trees) and Phase-3 inference is `O(α·N·T)` with
α ≈ 0.2 the alert fraction — i.e. the second stage adds only a sublinear
constant-factor overhead on top of the first.

---

## 4. Discussion and Insights

### 4.1 False-positive reduction — the mechanism

Stage 1 inspects a **single packet** at a time, so any momentarily unusual
benign packet (a large transfer, an odd inter-arrival gap) can trip it — hence
10,908 false alarms. Stage 2 looks at the **whole flow** the packet belongs to:
aggregated over a flow, benign behaviour is unmistakably benign (steady packet
sizes, normal totals, ordinary flag mix), so the supervised classifier confidently
rejects those alerts. The cascade's AND-logic means an alert survives only if it
is anomalous *both* in isolation *and* in aggregate — which is exactly the
profile of a real attack. This is why precision jumps from 0.12 to 0.993 while
recall is untouched.

### 4.2 Flow-based advantages (hypothesis test)

Running the **same** unsupervised ensemble on flow features vs packet features:

| | Packet-level | Flow-level |
|---|---|---|
| Unsupervised AUC-ROC | 0.917 | **0.982** |

The hypothesis that **flow features are more separable in the unsupervised
setting holds** (AUC 0.982 > 0.917). Aggregation denoises: a per-packet length
is noisy, but a flow's mean/peak/spread of lengths is a stable signature. This
is also why the supervised stage — which is *built* on flow features — reaches
near-perfect F1.

### 4.3 Attack-type analysis — who benefits, and why

Comparing per-attack detection at the packet level (Stage 1) vs the flow-level
*unsupervised* detector reveals a clean division of labour:

| Attack | Packet unsup. | Flow unsup. | Who wins |
|---|---|---|---|
| DoS-HTTP Flood | 0.997 | 0.694 | **packets** — *every packet* is anomalous (tiny, high-rate) |
| DDoS-HTTP Flood | 0.988 | 0.753 | **packets** — same volumetric signature |
| DNS Spoofing | 0.969 | 0.120 | **packets** — protocol/port anomaly is obvious per packet; the short flows are easy to miss in aggregate |
| Brute Force | 0.923 | **0.994** | **flow** — the repeated-handshake pattern is a flow-level property |
| **XSS** | **0.578** | **0.664** | **flow** — stealthy; the signal only emerges once the session is aggregated |

**The two views are complementary, and that is the whole point.** Volumetric
floods and DNS spoofing are trivially caught packet-by-packet (each malicious
packet is individually weird), so aggregating them actually *dilutes* the signal
at a strict threshold. The stealthier attacks (XSS, brute force) are the
opposite: no single packet looks malicious, but the flow does. **XSS and brute
force benefit most from the flow/two-stage view**, while floods and DNS are
already solved at the packet level. A hybrid that uses *both* views is therefore
strictly better than either alone — the core argument for the two-stage design.

### 4.4 Did the engineered flow-anomaly feature help?

We trained the supervised model **with and without** the Phase-3.1 flow-anomaly
score as an extra feature. The difference was **negligible** (F1 0.9945 vs
0.9944 — a one-row tie-break on the validation set, after which the pipeline
kept the marginally-better augmented model). The honest reason it cannot help
much here: the base flow features are already so separable that an extra
unsupervised score is largely redundant. We report this near-null result rather
than hide it — on noisier real data, where the supervised model is *not*
near-perfect, such a feature is far more likely to earn its place.

### 4.5 Flow-length analysis

Does a longer flow (more packets / longer duration) make an attack easier to
detect? Measured on the flow-level unsupervised detector at a *selective*
operating point (`results/metrics/flow_length_analysis_demo.json`):

* **Duration vs detection:** Pearson r = **0.29, p ≈ 3e-8** — longer flows are
  significantly more detectable.
* **Packet count vs detection (overall):** r = 0.18, p ≈ 1e-3 — positive but
  weaker, because it is *confounded* by attack type (floods carry many packets
  yet are detected differently than long stealthy sessions).
* **Within-attack-type** (removes the confound): the effect is concentrated in
  the **non-volumetric** attacks — **XSS r = 0.69 (p ≈ 2e-9)** and
  **DNS Spoofing r = 0.72** — whereas for the floods packet count is essentially
  irrelevant (DDoS r ≈ 0.14, DoS r ≈ 0.01) because they are already obvious.

**Conclusion:** flow length helps *most where it matters* — the stealthier
attacks whose evidence accumulates over the session. For volumetric attacks the
first packet already gives the game away. (Figure: `flow_length_analysis.svg`.)

### 4.6 Cluster structure (DoS/DDoS view)

The k-means cluster-composition plot (`phase2_clusters.svg`) shows benign
traffic concentrated in a few dense clusters, while the HTTP-flood DoS/DDoS
packets collapse into their own small, near-pure high-reconstruction-error
clusters — they sit far from the benign manifold. DoS and DDoS are
near-indistinguishable *from each other* in this unsupervised view (same HTTP
flood signature; they differ mainly in source diversity, which is an IP-level
property we intentionally exclude), but both are cleanly separated from benign —
which is all Stage 1 needs.

---

## 5. Conclusion and Future Work

### 5.1 Key findings

* A two-stage **unsupervised → supervised** cascade cut packet-level false
  positives by **99.9 % (10,908 → 10)** while **preserving recall**, lifting
  precision from 0.12 to 0.993 and overall accuracy to 0.997. The gain is
  statistically overwhelming (McNemar p ≈ 0; it corrects 10,898 errors and
  introduces 1).
* The second stage runs on only **~20 %** of traffic (the flagged subset), so
  the accuracy gain comes with low marginal cost.
* **Flow features are more separable than packet features** in the unsupervised
  setting (AUC 0.982 vs 0.917), confirming the brief's hypothesis.
* Packet- and flow-level views are **complementary**: packets dominate on
  volumetric floods and DNS spoofing, flows dominate on stealthy XSS and brute
  force — justifying a hybrid.
* Flow length correlates with detectability **specifically for the
  non-volumetric attacks** (within-type r ≈ 0.69 for XSS, 0.72 for DNS spoofing).

### 5.2 Limitations

* Results are on a **schema-faithful synthetic stand-in** for CIC IoT-DIAD 2024
  (the real host was unreachable in our environment). The synthetic data is
  cleaner than real captures, which is why the supervised stage is near-perfect;
  on real data we would expect lower absolute numbers and a larger marginal
  contribution from the engineered features and the second stage.
* The cascade's recall is **upper-bounded by Stage 1** (AND-logic). If Stage 1
  misses an attack, the system misses it; the wide alert budget mitigates but
  does not eliminate this.
* DoS vs DDoS are not separated in the unsupervised view because we exclude
  IP-level identifiers (by design, to avoid host memorisation).

### 5.3 Future work

* Run the identical pipeline on the **real CIC CSVs** (drop them in `data/`).
* Replace the AND-cascade with a **learned meta-combiner** (stacking Stage-1
  score + Stage-2 probability) so Stage 2 can also *recover* attacks Stage 1
  missed, lifting the recall ceiling.
* Add **graph / GNN** features over the IP–port communication graph to recover
  the DoS-vs-DDoS distinction (source-diversity is a graph property).
* Calibrate the alert budget online to a target analyst workload, and add
  per-attack thresholds.

---

### Appendix — Reproducing this report

```bash
pip install -r requirements.txt
python run.py --scale full          # Sections 3–4 headline numbers
python -m src.analysis --scale demo # Section 4.5 flow-length analysis
python tests/test_basic.py          # sanity checks
```
Every run is deterministic given `--seed`; metrics are written to
`results/metrics/` and figures to `results/figures/`.
