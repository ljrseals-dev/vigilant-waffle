"""
Regression fixture for vme_v3.py.

Locks in the metrics verified by actually running vme_v3.py (seed=42,
160 notes), so that any future change to the resolver, raw pitch
source, or chord model is caught if it shifts these numbers.

These are NOT aspirational targets (see README for those) -- they are
the actual measured output of the current code, confirmed reproducible
across repeated runs before being written here.
"""
import numpy as np
from vme_v3 import (
    DownwashResolver, generate_raw_pitch, best_key_24,
)
import random
import torch


def generate_pitches(seed=42, n_notes=160):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    resolver = DownwashResolver()
    pitches, last_raw = [], None
    for _ in range(n_notes):
        raw = generate_raw_pitch(last_raw)
        last_raw = raw
        resolver.maybe_change_chord()
        pitches.append(resolver.fit_pitch_to_chord(raw))
    return pitches


def compute_metrics(pitches):
    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    stepwise = 100.0 * sum(1 for i in intervals if i <= 2) / len(intervals)
    repeats = 100.0 * sum(1 for i in intervals if i == 0) / len(intervals)
    mean_int = float(np.mean(intervals))
    key_name, key_corr = best_key_24(pitches)
    return stepwise, repeats, mean_int, key_name, key_corr


def test_reproducibility():
    """Same seed must produce byte-identical pitch sequences."""
    p1 = generate_pitches(seed=42)
    p2 = generate_pitches(seed=42)
    assert p1 == p2, "Generation is not reproducible for a fixed seed"


def test_locked_metrics():
    """Locks in the verified metrics for seed=42, 160 notes."""
    pitches = generate_pitches(seed=42)
    stepwise, repeats, mean_int, key_name, key_corr = compute_metrics(pitches)

    assert abs(stepwise - 67.3) < 0.5, f"Stepwise% drifted: {stepwise}"
    assert abs(repeats - 1.3) < 0.5, f"Repeat% drifted: {repeats}"
    assert abs(mean_int - 1.91) < 0.05, f"Mean interval drifted: {mean_int}"
    assert key_name == "A minor", f"Best-fit key changed: {key_name}"
    assert abs(key_corr - 0.741) < 0.01, f"Key correlation drifted: {key_corr}"


def test_known_limitation_documented():
    """Sanity check that the known gap (mean interval below 2.5) is
    still present -- if this ever passes with mean_int >= 2.5, the
    README's Known Limitation section needs to be updated, not just
    this test."""
    pitches = generate_pitches(seed=42)
    _, _, mean_int, _, _ = compute_metrics(pitches)
    assert mean_int < 2.5, (
        "Mean interval now exceeds 2.5 -- if achieved without breaking "
        "stepwise% or key correlation, update README.md's Known "
        "Limitation section, this represents solving the Pareto problem."
    )


if __name__ == "__main__":
    test_reproducibility()
    print("test_reproducibility: PASSED")
    test_locked_metrics()
    print("test_locked_metrics: PASSED")
    test_known_limitation_documented()
    print("test_known_limitation_documented: PASSED")
