# Starter — two-stage IDS (Phase 2 + Phase 3)

Self-contained. Everything needed is in this folder.

## Files
- `common.py`  — load CSVs, sample, aggregate flows, metrics
- `models.py`  — the `Detector` (unsupervised) and `Classifier` (supervised)
- `unsupervised.py` — Phase 2: trains the anomaly detector on packet data
- `supervised.py`   — Phase 3: trains the classifier on flow data

## Data layout
Put the CIC IoT-DIAD CSVs here (HTTP-Flood files for DoS/DDoS):
```
data/packet/<Benign|BruteForce|DDoS|DoS|Spoofing|Web-Based>/*.csv
data/flow/<Benign|BruteForce|DDoS|DoS|Spoofing|Web-Based>/*.csv
```
The label is read from the folder name; `Mirai`/`Recon` are ignored.

## Run
```bash
pip install -r requirements.txt
python unsupervised.py     # Phase 2
python supervised.py       # Phase 3
```
Each saves a bundle to `artifacts/` for the cascade step later.

Big files / low RAM? cap rows per file:
```bash
IDS_MAX_ROWS_PER_FILE=400000 python unsupervised.py
```
