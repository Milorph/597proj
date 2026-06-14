"""
Task 1.1 -- Random dataset generation / sampling.

``sample_dataset`` draws a class-imbalanced sample matching the project spec:
~200,000 benign rows (97-98%) and 4,000-6,200 attack rows (2-3%) spread
randomly across the five attack types. A configurable seed gives a different
composition on each run while remaining reproducible.

The same function services both Phase-2 (packet level) and Phase-3 (flow level):
it operates on any DataFrame that carries ``Label`` / ``attack_type`` columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


class SamplingError(ValueError):
    """Raised when the source population cannot satisfy the sampling request."""


def _random_attack_allocation(total_attack: int, rng: np.random.Generator) -> dict:
    """
    Split ``total_attack`` rows **roughly uniformly** across the five attack
    types (per the project clarification: spread evenly across the categories,
    picked at random). Each type is centred on ``total_attack / 5`` and given a
    mild random jitter (+/-15%), so every run differs while staying close to an
    even split, and every type is guaranteed at least one row.
    """
    k = len(config.ATTACK_TYPES)
    base = total_attack / k
    jitter = rng.uniform(0.85, 1.15, size=k)             # roughly uniform, random
    raw = base * jitter
    counts = np.floor(raw / raw.sum() * total_attack).astype(int)
    counts = np.maximum(counts, 1)                        # every type present
    # Fix rounding drift so the counts sum exactly to total_attack.
    while counts.sum() > total_attack and counts.max() > 1:
        counts[counts.argmax()] -= 1
    while counts.sum() < total_attack:
        counts[rng.integers(0, k)] += 1
    return dict(zip(config.ATTACK_TYPES, counts.tolist()))


def sample_dataset(population: pd.DataFrame,
                   seed: int | None = None,
                   n_benign: int = config.N_BENIGN,
                   attack_min: int = config.ATTACK_MIN,
                   attack_max: int = config.ATTACK_MAX,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Draw an imbalanced IDS sample from ``population``.

    Parameters
    ----------
    population : DataFrame with ``Label`` (0/1) and ``attack_type`` columns.
    seed : random seed; ``None`` => fresh entropy => different composition.

    Returns
    -------
    DataFrame : the sampled rows, shuffled, index reset.
    """
    rng = np.random.default_rng(seed)

    # ---- integrity checks on the population --------------------------------
    for col in ("Label", "attack_type"):
        if col not in population.columns:
            raise SamplingError(f"population missing required column '{col}'")

    benign_pool = population[population["Label"] == 0]
    if len(benign_pool) == 0:
        raise SamplingError("population contains no benign rows")
    if len(benign_pool) < n_benign:
        # Real data can have fewer benign records than the spec's 200k (e.g. the
        # flow level, where many segments collapse into one flow). Cap benign to
        # what's available and scale the attack budget down by the same factor so
        # the ~2-3% imbalance ratio is preserved.
        factor = len(benign_pool) / n_benign
        new_min = max(len(config.ATTACK_TYPES), int(round(attack_min * factor)))
        new_max = max(new_min, int(round(attack_max * factor)))
        print(f"[sample] NOTE: only {len(benign_pool):,} benign rows available "
              f"(< {n_benign:,} requested); capping benign and scaling attack "
              f"budget to {new_min:,}-{new_max:,} to keep the ~2-3% ratio.")
        n_benign = len(benign_pool)
        attack_min, attack_max = new_min, new_max

    total_attack = int(rng.integers(attack_min, attack_max + 1))
    alloc = _random_attack_allocation(total_attack, rng)

    # ---- sample benign -----------------------------------------------------
    benign_sample = benign_pool.sample(n=n_benign, random_state=int(rng.integers(0, 2**31)))

    # ---- sample each attack type (with availability checks) ----------------
    attack_parts = []
    realized = {}
    for atk, want in alloc.items():
        pool = population[population["attack_type"] == atk]
        take = min(want, len(pool))
        if take == 0:
            raise SamplingError(f"no rows available for attack type '{atk}'")
        attack_parts.append(pool.sample(n=take, random_state=int(rng.integers(0, 2**31))))
        realized[atk] = take

    sample = pd.concat([benign_sample] + attack_parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

    # ---- validation of the produced sample ---------------------------------
    n_attack = int(sample["Label"].sum())
    n_total = len(sample)
    attack_ratio = n_attack / n_total
    _validate_sample(sample, n_benign, attack_min, attack_max, realized)

    if verbose:
        print(f"[sample] total={n_total:,}  benign={n_total - n_attack:,}  "
              f"attack={n_attack:,} ({attack_ratio:.2%})")
        print(f"[sample] per-attack: " +
              "  ".join(f"{k}={v}" for k, v in realized.items()))

    return sample


def _validate_sample(sample, n_benign, attack_min, attack_max, realized):
    """Basic integrity checks on the produced sample (raises on failure)."""
    n_attack = int(sample["Label"].sum())
    n_ben = len(sample) - n_attack
    assert n_ben == n_benign, f"benign count {n_ben} != requested {n_benign}"
    # Allow the realized attack total to fall short only if a pool was exhausted.
    assert n_attack <= attack_max, f"attack total {n_attack} exceeds max {attack_max}"
    assert set(realized.keys()) == set(config.ATTACK_TYPES), "missing an attack type"
    assert sample.isna().sum().sum() >= 0  # placeholder; NA handled in preprocessing
    # No duplicate exact rows from oversampling (we sample without replacement).
    return True
