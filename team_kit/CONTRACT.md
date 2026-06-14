# Team Contract — freeze this at kickoff, then work independently

Three groups, **no contact needed until integration**. Everyone develops on the
synthetic data (`contract.get_*`) and mocks. Swap in real models at the end.

## Groups & ownership
| Group | People | Implements | Runs standalone with |
|---|---|---|---|
| **Unsupervised (Phase 2)** | 2 | `AnomalyDetector` + own packet preprocessing | real synthetic data |
| **Supervised (Phase 3)** | 2 | `SignatureClassifier` + own flow preprocessing | real synthetic data |
| **Cascade + Eval + Report** | 3 | two-stage glue, metrics, report, README, video | **mocks** (no waiting) |

## The 4 frozen agreements
1. **Schema** — column names come from `config.py`. Don't rename.
2. **`flow_id`** — the join key, built by the data layer (canonical
   `sortedendpoint-endpoint`). Never rebuild it yourself.
3. **Artifact** — each model group ships a `contract.Bundle(kind, preprocessor,
   model, extra)` saved with `contract.save_bundle()` to
   `team_kit/artifacts/{unsupervised,supervised}.joblib`. **The fitted
   preprocessor ships *inside* the bundle** — the cascade never re-fits.
4. **Metrics** — use `contract.binary_metrics()` / `per_attack_detection_rate()`
   so all numbers are comparable.

## Interfaces (the only functions other groups rely on)
```python
class AnomalyDetector:      # Phase 2 — NO labels in fit()
    def fit(self, X): ...
    def anomaly_score(self, X): ...     # higher = more anomalous
    def predict(self, X): ...           # 1 = alert

class SignatureClassifier:  # Phase 3 — labels in fit(), not at test
    def fit(self, X, y): ...
    def predict_proba(self, X): ...     # P(attack)
    def predict(self, X, threshold=0.5): ...

class FeatureTransformer:   # each group's own preprocessing
    feature_names: list
    def fit_transform(self, df): ...
    def transform(self, df): ...
```

## Day-1 commands (each group, in parallel)
```bash
python -m team_kit.group_unsupervised_starter   # Phase 2 group
python -m team_kit.group_supervised_starter     # Phase 3 group
python -m team_kit.group_cascade_starter        # Cascade group (uses mocks)
```

## Integration day
1. Phase-2 & Phase-3 groups commit their `.joblib` bundles to `team_kit/artifacts/`.
2. Cascade group re-runs `group_cascade_starter` — it auto-detects the real
   bundles and replaces the mocks. Done.

## Rules that prevent silent breakage
- **Never re-fit a preprocessor at the boundary** — use the one in the bundle.
- **Phase 2 must not read labels in `fit`** (labels are eval-only).
- **Keep the dataset out of git** (`data/` is gitignored).
- One git branch per group; commit under your own identity.
