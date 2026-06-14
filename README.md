# Multi-Stage IoT Network Intrusion Detection
### ECE 597 Capstone — Unsupervised + Supervised Two-Stage IDS

A two-stage intrusion-detection system for IoT networks built on the
**CIC IoT-DIAD 2024** dataset:

1. **Stage 1 — Anomaly-based IDS (unsupervised, packet level).** An
   autoencoder + Isolation Forest + K-means ensemble flags suspicious packets
   without using labels. It deliberately casts a *wide net* (high recall, many
   false positives).
2. **Stage 2 — Signature-based IDS (supervised, flow level).** A gradient-boosted
   tree classifier re-checks every Stage-1 alert using aggregated flow features,
   removing the false positives while preserving detection.

The cascade keeps Stage-1's detection rate but cuts its false positives by
**~99–100%** in our experiments.

---

## 1. Quick start

```bash
# (1) install dependencies
pip install -r requirements.txt

# (2) run the whole pipeline (synthetic data, ~30 s)
python run.py --scale demo

# (3) run at the full spec scale (200k benign / 4k–6.2k attack, a few minutes)
python run.py --scale full

# tiny smoke run (CI / quick check)
python run.py --scale smoke

# different random composition
python run.py --seed 7

# skip the optional Task 3.1 flow-level unsupervised stage
python run.py --no-flow-unsup
```

Outputs:
- `results/metrics/results_<scale>.json` — every metric the brief requires.
- `results/figures/*.png` — confusion matrices, ROC/PR curves, per-attack bars,
  anomaly-score distribution, and the K-means cluster-composition plot.

Run the tests:

```bash
python tests/test_basic.py        # or: pytest tests/
```

---

## 2. Data

> **Important.** The real CIC IoT-DIAD 2024 CSVs are large and must be
> downloaded separately. This project was developed in an environment whose
> network policy blocks the dataset host, so the code ships with a
> **schema-faithful synthetic generator** (`src/data_synth.py`) that stands in
> for the download. The synthetic path lets the *entire* pipeline run and
> produce every deliverable. **The downstream code is identical for real and
> synthetic data** — drop the real files in and nothing else changes.

### Using the real dataset

1. Download both the **packet-level** and **flow-level** files from
   <http://cicresearch.ca/IOTDataset/CIC%20IoT-IDAD%20Dataset%202024/>
   (dataset info: <https://www.unb.ca/cic/datasets/iot-diad-2024.html>).
2. Place / concatenate them as:
   ```
   data/packet_level.csv
   data/flow_level.csv
   ```
   (Override paths with the `IDS_PACKET_CSV` / `IDS_FLOW_CSV` env vars, or the
   parent directory with `IDS_DATA_DIR`.)
3. Run `python run.py --scale full`. The loader auto-detects the real files;
   `src/data_loader._normalize_labels` maps the published label strings onto the
   five canonical attack classes and derives the binary `Label`.

Attacks used (per the brief; flow data for the TCP-Flood variants is not in this
release, so the **HTTP-Flood** variants are used for DoS and DDoS):

`DDoS-HTTP_Flood`, `DoS-HTTP_Flood`, `DNS_Spoofing`, `XSS`, `Brute_Force`.

### The synthetic generator

`src/data_synth.py` produces two coherent tables linked by a
`flow_id = src_ip-dst_ip-src_port-dst_port`:

- a **packet-level** table (one row per packet), and
- a **flow-level** table (one row per *2-minute flow segment*) — flows longer
  than 120 s are split into successive segments that share a `flow_id`, exactly
  as the brief describes, so the Phase-3 group-by-`flow_id` aggregation has real
  work to do.

Each class has a distinct-but-overlapping signature rooted in network intuition
(e.g. HTTP floods = many tiny high-rate packets from many/few sources; XSS =
large script payloads, low rate; brute force = many short SYN/RST attempts;
DNS spoofing = small UDP/53 packets). Flow-level *aggregate* features encode the
class signal more cleanly than noisy per-packet features — which is what lets us
test the brief's hypothesis that flow features help.

---

## 3. Project layout

```
config.py                 # paths, attack taxonomy, sampling spec, hyperparams
run.py                    # CLI entry point
src/
  data_synth.py           # schema-faithful CIC IoT-DIAD 2024 synthetic generator
  data_loader.py          # real-or-synthetic loader + label normalisation
  sampling.py             # Task 1.1 imbalanced sampler (packet & flow)
  preprocessing.py        # Task 1.2 clean / impute / clip / scale
  flow_aggregation.py     # group 2-min segments by flow_id (per-feature agg)
  unsupervised.py         # Phase 2: autoencoder + IsolationForest + KMeans
  supervised.py           # Phase 3: XGBoost / HistGBDT signature classifier
  evaluate.py             # metrics + all figures
  pipeline.py             # end-to-end orchestration
tests/test_basic.py       # sampling / aggregation / preprocessing checks
report/REPORT.md          # full written report
results/                  # generated metrics + figures
```

---

## 4. How the pipeline maps to the brief

| Brief task | Where |
|---|---|
| 1.1 Random dataset generation (200k benign, 4k–6.2k attack, random mix) | `src/sampling.py::sample_dataset` |
| 1.2 Preprocessing (missing/outliers/scaling/selection) | `src/preprocessing.py::Preprocessor` |
| 2.1 Packet-level unsupervised detection (AE + k-means) | `src/unsupervised.py` |
| 2.2 Threshold selection + FP/FN analysis | `pipeline.py` (alert budget + FPR-targeted threshold) |
| 3.x Flow segmentation → one record per flow | `src/flow_aggregation.py` |
| 3.1 Flow-level unsupervised + packet-vs-flow comparison | `pipeline.py` (`task3_1_*`) |
| 3.2 Flow-based feature engineering / second flow sample | `pipeline.py` + `flow_aggregation.py` |
| 3.3 Supervised signature IDS, with/without flow-anomaly feature | `src/supervised.py` + `pipeline.py` |
| Evaluation (all metrics, ROC/PR/CM, per-attack, McNemar, timing) | `src/evaluate.py` + `pipeline.py` |

See `report/REPORT.md` for the full methodology, results and discussion.

---

## 5. Reproducibility

Every run is deterministic given `--seed` (the synthetic population, the
sampling, and all models are seeded). The sampler draws a *different attack
composition* per seed while keeping the 200k-benign / 2–3%-attack proportions,
satisfying the "different composition on each run" requirement.
