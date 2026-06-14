"""
Central configuration for the two-stage IoT IDS project.

All tunable constants, file paths, and the canonical list of attack types live
here so that every module shares one source of truth.

NOTE ON DATA SOURCE
-------------------
The real CIC IoT-DIAD 2024 dataset must be downloaded from
    http://cicresearch.ca/IOTDataset/CIC%20IoT-IDAD%20Dataset%202024/
and the packet-level / flow-level CSVs placed under ``DATA_DIR`` (see README).
If those files are not present, the pipeline falls back to a schema-faithful
synthetic generator (``src/data_synth.py``) so the full system remains runnable
and reproducible without the gated download. The downstream code path is
identical for real and synthetic data.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("IDS_DATA_DIR", os.path.join(ROOT_DIR, "data"))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# Real-dataset inputs. Two ways to provide them (either works):
#   (a) single concatenated CSVs:  data/packet_level.csv , data/flow_level.csv
#   (b) folders of raw CIC files:   data/packet/**.csv  ,  data/flow/**.csv
#       -> the loader globs every CSV under the folder and infers the attack
#          label from the sub-folder name (Benign/BruteForce/DDoS/DoS/...),
#          so you can just drop the downloaded files in without merging them.
PACKET_CSV = os.environ.get("IDS_PACKET_CSV", os.path.join(DATA_DIR, "packet_level.csv"))
FLOW_CSV = os.environ.get("IDS_FLOW_CSV", os.path.join(DATA_DIR, "flow_level.csv"))
PACKET_DIR = os.environ.get("IDS_PACKET_DIR", os.path.join(DATA_DIR, "packet"))
FLOW_DIR = os.environ.get("IDS_FLOW_DIR", os.path.join(DATA_DIR, "flow"))
# Optional safety cap on rows read per raw CSV (real packet files are huge).
# Set IDS_MAX_ROWS_PER_FILE=500000 etc. to bound memory; default = read all.
MAX_ROWS_PER_FILE = int(os.environ["IDS_MAX_ROWS_PER_FILE"]) if os.environ.get("IDS_MAX_ROWS_PER_FILE") else None

# --------------------------------------------------------------------------- #
# Attack taxonomy (per project spec, Section 3)
# --------------------------------------------------------------------------- #
BENIGN_LABEL = "Benign"
ATTACK_TYPES = [
    "DDoS-HTTP_Flood",
    "DoS-HTTP_Flood",
    "DNS_Spoofing",
    "XSS",
    "Brute_Force",
]
ALL_CLASSES = [BENIGN_LABEL] + ATTACK_TYPES

# --------------------------------------------------------------------------- #
# Sampling specification (Task 1.1)
# --------------------------------------------------------------------------- #
N_BENIGN = 200_000          # benign rows per sampled dataset (~97-98%)
ATTACK_MIN = 4_000          # lower bound on total attack rows (~2-3%)
ATTACK_MAX = 6_200          # upper bound on total attack rows

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
DEFAULT_SEED = 42

# --------------------------------------------------------------------------- #
# Synthetic-data scale (only used when real CSVs are absent)
# --------------------------------------------------------------------------- #
# Size of the synthetic *population* the sampler draws from. Kept well above the
# sample size so random sampling is meaningful, but small enough to run on a
# laptop in a few minutes.
SYNTH_BENIGN_POP = 320_000
SYNTH_ATTACK_POP_PER_TYPE = 9_000

# --------------------------------------------------------------------------- #
# Phase 2 (unsupervised) defaults
# --------------------------------------------------------------------------- #
AE_HIDDEN = (32, 12, 32)    # MLP autoencoder bottleneck architecture
AE_MAX_ITER = 60
KMEANS_K = 8                # over-clustering, clusters mapped to benign/anomaly
ISO_FOREST_ESTIMATORS = 200

# Threshold selection: target benign false-positive rate when picking the
# reconstruction-error / anomaly-score cut-off.
TARGET_FPR = 0.05

# Phase-3 cascade: fraction of true attacks the confirmation stage must keep.
# The cascade rejects an alert only when the flow model is confident it is benign,
# so detection (recall) is preserved while false positives are removed.
TARGET_CASCADE_RECALL = 0.95

# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.30
VAL_SIZE = 0.20             # fraction of the train split held out for validation


def ensure_dirs() -> None:
    """Create the output directory tree if it does not yet exist."""
    for d in (DATA_DIR, RESULTS_DIR, FIG_DIR, METRICS_DIR):
        os.makedirs(d, exist_ok=True)
