"""
Lightweight self-checks for the core building blocks. Runnable with either
``python tests/test_basic.py`` or ``pytest``. Uses a tiny synthetic population
so it finishes in a couple of seconds.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import data_synth, flow_aggregation
from src.preprocessing import Preprocessor
from src.sampling import sample_dataset, SamplingError


def _small_pop():
    return data_synth.generate(seed=1, benign_pop=12_000, attack_pop_per_type=600)


def test_sampling_proportions():
    packets, _ = _small_pop()
    s = sample_dataset(packets, seed=7, n_benign=8_000,
                       attack_min=160, attack_max=240, verbose=False)
    n_attack = int(s["Label"].sum())
    assert (s["Label"] == 0).sum() == 8_000              # exact benign count
    assert 160 <= n_attack <= 240                         # attack budget respected
    ratio = n_attack / len(s)
    assert 0.015 <= ratio <= 0.035                        # ~2-3% imbalance
    # every attack type represented
    present = set(s.loc[s["Label"] == 1, "attack_type"].unique())
    assert present == set(config.ATTACK_TYPES)
    print("test_sampling_proportions OK")


def test_sampling_is_random():
    packets, _ = _small_pop()
    a = sample_dataset(packets, seed=1, n_benign=8_000, attack_min=160,
                       attack_max=240, verbose=False)
    b = sample_dataset(packets, seed=2, n_benign=8_000, attack_min=160,
                       attack_max=240, verbose=False)
    # Different seeds => different composition (attack totals or makeup differ).
    assert not a.equals(b)
    print("test_sampling_is_random OK")


def test_sampling_validation_raises():
    packets, _ = _small_pop()
    try:
        sample_dataset(packets, seed=0, n_benign=10_000_000, verbose=False)
    except SamplingError:
        print("test_sampling_validation_raises OK")
        return
    raise AssertionError("expected SamplingError for impossible benign request")


def test_flow_aggregation_one_record_per_flow():
    _, flows = _small_pop()
    unified = flow_aggregation.aggregate_flows(flows, verbose=False)
    assert unified["flow_id"].is_unique                   # one record per flow
    assert (unified["n_segments"] >= 1).all()
    # additive features must aggregate by sum (>= any single segment's value)
    assert "total_packets" in unified.columns
    print("test_flow_aggregation_one_record_per_flow OK")


def test_preprocessor_no_nan_inf():
    packets, _ = _small_pop()
    s = sample_dataset(packets, seed=3, n_benign=8_000, attack_min=160,
                       attack_max=240, verbose=False)
    X = Preprocessor().fit_transform(s)
    assert np.isfinite(X).all()                           # no NaN/Inf leaks through
    assert X.shape[0] == len(s)
    print("test_preprocessor_no_nan_inf OK")


if __name__ == "__main__":
    test_sampling_proportions()
    test_sampling_is_random()
    test_sampling_validation_raises()
    test_flow_aggregation_one_record_per_flow()
    test_preprocessor_no_nan_inf()
    print("\nALL TESTS PASSED")
