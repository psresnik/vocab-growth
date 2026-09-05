"""
Tests for analysis.py, plus the substantive claims the whole project is about.

The tests at the bottom of this file are the interesting ones: they check that
the model actually behaves the way the theory says it should.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import analysis
import config
import difficulty
import simulation


def _run(distribution_name, n_words=4000, n_steps=9000, seed=101):
    """Helper: run one simulation and return its growth curve."""
    thresholds = difficulty.make_thresholds(
        distribution_name, n_words, config.THRESHOLD_MEAN, config.THRESHOLD_SD, seed)
    steps = max(n_steps, int(np.percentile(thresholds, 99)) + 10)
    result = simulation.simulate_parallel_learning(thresholds, steps)
    return result["known_after_step"]


# ---------------------------------------------------------------------------
# The metric functions themselves
# ---------------------------------------------------------------------------

def test_words_learned_in_window_counts_correctly():
    """Window counts are simple differences of the curve."""
    curve = np.array([0, 1, 3, 6, 10, 15])
    assert analysis.words_learned_in_window(curve, 0, 5) == 15
    assert analysis.words_learned_in_window(curve, 2, 4) == 7
    assert analysis.words_learned_in_window(curve, 3, 3) == 0


def test_window_past_end_is_rejected():
    """Asking for a window past the end of the curve is an error, not a guess."""
    curve = np.array([0, 1, 2])
    try:
        analysis.words_learned_in_window(curve, 0, 99)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a ValueError for an out-of-range window.")


def test_backwards_window_is_rejected():
    """A window that ends before it starts is an error."""
    curve = np.array([0, 1, 2, 3])
    try:
        analysis.words_learned_in_window(curve, 3, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a ValueError for a backwards window.")


def test_learning_rate_on_a_straight_line_is_constant():
    """A curve growing by 1 per step has a flat rate."""
    curve = np.arange(0, 101)
    rates = analysis.learning_rate(curve, window=10)
    assert np.all(rates == 10)


def test_rate_change_on_a_straight_line_is_zero():
    """A constant rate means no acceleration."""
    curve = np.arange(0, 101)
    changes = analysis.rate_change(curve, window=10)
    assert np.all(changes == 0)


def test_peak_rate_step_finds_the_fast_part():
    """The peak should land where the curve is steepest."""
    # Flat, then steep, then flat again.
    curve = np.concatenate([
        np.zeros(100, dtype=int),
        np.arange(0, 100, dtype=int),
        np.full(100, 99, dtype=int),
    ])
    peak = analysis.peak_rate_step(curve, window=10)
    assert 100 <= peak <= 200


def test_step_reaching_finds_first_crossing():
    """step_reaching returns the first step at or above the target."""
    curve = np.array([0, 0, 2, 5, 5, 9])
    assert analysis.step_reaching(curve, 1) == 2
    assert analysis.step_reaching(curve, 5) == 3
    assert analysis.step_reaching(curve, 100) == -1


# ---------------------------------------------------------------------------
# The substantive claims
# ---------------------------------------------------------------------------

def test_gaussian_difficulty_produces_acceleration():
    """The central claim: bell-shaped difficulty gives an accelerating curve."""
    curve = _run("gaussian")
    summary = analysis.describe_growth(curve, config.RATE_WINDOW)
    assert summary["accelerates"]
    # The peak should be many times faster than the start, not marginally so.
    assert summary["acceleration_ratio"] > 3.0


def test_gaussian_difficulty_also_decelerates_at_the_end():
    """Acceleration is followed by deceleration once easy words run out."""
    curve = _run("gaussian")
    summary = analysis.describe_growth(curve, config.RATE_WINDOW)
    assert summary["decelerates_at_end"]


def test_peak_learning_is_near_the_mean_difficulty():
    """Learning should be fastest around the average time-to-acquisition."""
    curve = _run("gaussian")
    peak = analysis.peak_rate_step(curve, config.RATE_WINDOW)
    assert abs(peak - config.THRESHOLD_MEAN) < 3 * config.THRESHOLD_SD


def test_exponential_difficulty_does_not_accelerate():
    """The negative control.

    With an exponential difficulty distribution there are MANY easy words and
    few hard ones, which is the opposite of the pattern that causes
    acceleration. Growth should slow down from the start.
    """
    curve = _run("exponential")
    summary = analysis.describe_growth(curve, config.RATE_WINDOW)
    assert not summary["accelerates"]


def test_uniform_difficulty_is_roughly_linear():
    """A flat difficulty distribution gives near-straight-line growth.

    The peak rate should be barely above the starting rate, so this does not
    count as acceleration.
    """
    curve = _run("uniform")
    summary = analysis.describe_growth(curve, config.RATE_WINDOW)
    assert not summary["accelerates"]
    assert summary["acceleration_ratio"] < 1.5


def test_multifactor_difficulty_accelerates_like_a_gaussian():
    """Summing many skewed factors gives the same behaviour as assuming a Gaussian.

    This matters: you do not have to assume a bell-shaped difficulty
    distribution, you get one whenever difficulty has many causes.
    """
    curve = _run("multifactor")
    summary = analysis.describe_growth(curve, config.RATE_WINDOW)
    assert summary["accelerates"]
    assert summary["acceleration_ratio"] > 1.5


def test_lognormal_and_gamma_accelerate():
    """Other bell-ish distributions should behave like the Gaussian."""
    for name in ("lognormal", "gamma"):
        curve = _run(name)
        summary = analysis.describe_growth(curve, config.RATE_WINDOW)
        assert summary["accelerates"], name


def test_frequency_difficulty_accelerates():
    """Difficulty built from a realistic frequency distribution accelerates too."""
    import datasets
    frequency_data = datasets.get_frequencies("child", "data", 2000)
    thresholds = difficulty.thresholds_from_frequencies(
        frequency_data["frequencies"], config.FREQ_BASE, config.FREQ_SCALE)
    n_steps = int(thresholds.max() * 1.05) + 10
    result = simulation.simulate_parallel_learning(thresholds, n_steps)
    summary = analysis.describe_growth(result["known_after_step"],
                                       config.RATE_WINDOW)
    assert summary["accelerates"]


def test_sampling_model_accelerates():
    """Acceleration appears even when every word is equally hard."""
    import datasets
    frequency_data = datasets.get_frequencies("child", "data", 1000)
    probabilities = difficulty.sampling_probabilities_from_frequencies(
        frequency_data["frequencies"])
    n_steps = simulation.choose_sampling_horizon(probabilities, 10)
    result = simulation.simulate_sampled_learning(probabilities, 10, n_steps,
                                                  seed=53)
    summary = analysis.describe_growth(result["known_after_step"], 500)
    assert summary["accelerates"]


def test_acceleration_survives_both_benefit_and_cost():
    """Acceleration cannot be used as evidence for a helping mechanism.

    The curve accelerates whether learning a word helps with the next one or
    gets in the way, so seeing acceleration tells you nothing about which is
    happening.
    """
    thresholds = difficulty.make_gaussian_thresholds(
        4000, config.THRESHOLD_MEAN, config.THRESHOLD_SD, seed=59)
    for shift in (0.0, -0.1, +0.1):
        result = simulation.simulate_parallel_learning(thresholds, 8000, shift)
        summary = analysis.describe_growth(result["known_after_step"],
                                           config.RATE_WINDOW)
        assert summary["accelerates"], "shift=%s did not accelerate" % shift


def test_bimodal_difficulty_gives_two_spurts():
    """Two clusters of difficulty produce two separate bursts of learning."""
    curve = _run("bimodal", n_words=6000)
    rates = analysis.learning_rate(curve, config.RATE_WINDOW)
    # Count local peaks that are a decent fraction of the tallest one.
    tall_enough = rates > 0.3 * rates.max()
    n_peaks = 0
    for index in range(1, len(rates) - 1):
        if tall_enough[index] and rates[index] >= rates[index - 1] \
                and rates[index] > rates[index + 1]:
            n_peaks += 1
    assert n_peaks >= 2, "expected two spurts, found %d peaks" % n_peaks
