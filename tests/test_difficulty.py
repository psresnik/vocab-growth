"""Tests for difficulty.py -- building time-to-acquisition thresholds."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
import difficulty


def test_gaussian_has_requested_shape():
    """A large Gaussian sample should have close to the requested mean and sd."""
    thresholds = difficulty.make_gaussian_thresholds(20000, 4000.0, 680.0, seed=1)
    assert len(thresholds) == 20000
    assert abs(thresholds.mean() - 4000.0) < 20.0
    assert abs(thresholds.std() - 680.0) < 20.0


def test_same_seed_gives_same_thresholds():
    """Reproducibility: the same seed must give exactly the same numbers."""
    first = difficulty.make_gaussian_thresholds(500, 4000.0, 680.0, seed=42)
    second = difficulty.make_gaussian_thresholds(500, 4000.0, 680.0, seed=42)
    assert np.array_equal(first, second)


def test_different_seed_gives_different_thresholds():
    """Different seeds must not give identical results."""
    first = difficulty.make_gaussian_thresholds(500, 4000.0, 680.0, seed=1)
    second = difficulty.make_gaussian_thresholds(500, 4000.0, 680.0, seed=2)
    assert not np.array_equal(first, second)


def test_no_threshold_below_floor():
    """Thresholds must never drop below the configured minimum."""
    # A tiny mean with a big spread would otherwise produce negative values.
    thresholds = difficulty.make_gaussian_thresholds(5000, 10.0, 100.0, seed=3)
    assert thresholds.min() >= config.MIN_THRESHOLD


def test_every_registered_maker_works():
    """Every distribution in the registry must produce usable thresholds."""
    for name in difficulty.DIFFICULTY_MAKERS:
        thresholds = difficulty.make_thresholds(name, 1000, 4000.0, 680.0, seed=5)
        assert len(thresholds) == 1000, name
        assert np.all(thresholds >= config.MIN_THRESHOLD), name
        assert np.all(np.isfinite(thresholds)), name


def test_unknown_distribution_name_is_rejected():
    """Asking for a distribution that does not exist should raise a clear error."""
    try:
        difficulty.make_thresholds("banana", 10, 4000.0, 680.0, seed=1)
    except KeyError as error:
        assert "banana" in str(error)
        assert "gaussian" in str(error)  # the message lists valid options
    else:
        raise AssertionError("Expected a KeyError for an unknown distribution.")


def test_multifactor_sum_is_roughly_bell_shaped():
    """Adding many skewed factors should give something close to symmetric.

    This is the central limit theorem in action: the individual factors are
    strongly skewed exponentials, but their sum is not.
    """
    thresholds = difficulty.make_multifactor_thresholds(
        30000, 4000.0, 680.0, seed=7, n_factors=12)
    # Skewness of a symmetric distribution is 0. A single exponential has
    # skewness 2, so anything below 0.7 shows substantial symmetrising.
    centred = thresholds - thresholds.mean()
    skewness = (centred ** 3).mean() / (thresholds.std() ** 3)
    assert abs(skewness) < 0.7
    assert abs(thresholds.mean() - 4000.0) < 30.0


def test_bimodal_has_two_separated_groups():
    """The bimodal generator should produce two clearly separated clusters."""
    thresholds = difficulty.make_bimodal_thresholds(4000, 4000.0, 680.0, seed=9)
    # Split at the midpoint between the two intended centres and check both
    # halves are populated.
    midpoint = thresholds.mean()
    below = (thresholds < midpoint).sum()
    above = (thresholds >= midpoint).sum()
    assert below > 1000
    assert above > 1000


def test_thresholds_from_frequencies_ranks_correctly():
    """The most frequent word must get the smallest threshold."""
    frequencies = np.array([100.0, 10.0, 1.0])
    thresholds = difficulty.thresholds_from_frequencies(frequencies,
                                                        base=3000.0, scale=800.0)
    assert thresholds[0] == 3000.0            # most frequent gets the base
    assert thresholds[1] > thresholds[0]
    assert thresholds[2] > thresholds[1]
    # Equal log steps should give equal threshold steps.
    first_gap = thresholds[1] - thresholds[0]
    second_gap = thresholds[2] - thresholds[1]
    assert abs(first_gap - second_gap) < 1e-9


def test_thresholds_from_frequencies_rejects_zero():
    """Zero or negative frequencies have no logarithm and must be rejected."""
    try:
        difficulty.thresholds_from_frequencies(np.array([1.0, 0.0]), 3000.0, 800.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a ValueError for a zero frequency.")


def test_sampling_probabilities_sum_to_one():
    """Sampling probabilities must form a valid probability distribution."""
    frequencies = np.array([100.0, 50.0, 10.0, 1.0])
    probabilities = difficulty.sampling_probabilities_from_frequencies(frequencies)
    assert abs(probabilities.sum() - 1.0) < 1e-12
    assert np.all(probabilities > 0)
    # More frequent words must be more likely to be encountered.
    assert probabilities[0] > probabilities[1] > probabilities[2] > probabilities[3]
