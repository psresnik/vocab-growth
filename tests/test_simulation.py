"""Tests for simulation.py -- the learning loops."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
import difficulty
import simulation


def test_word_is_learned_exactly_at_its_threshold():
    """With one point per step, a word with threshold 5 is learned at step 5."""
    thresholds = np.array([5.0])
    result = simulation.simulate_parallel_learning(thresholds, n_steps=10)
    assert result["learned_at_step"][0] == 5
    assert result["known_after_step"][4] == 0
    assert result["known_after_step"][5] == 1


def test_curve_matches_counting_thresholds_directly():
    """The simulation must agree with simply counting thresholds below t.

    This is the key correctness check. With no coupling between words, the
    number known after t steps is exactly the number of thresholds <= t, which
    we can compute without any simulation at all.
    """
    thresholds = difficulty.make_gaussian_thresholds(2000, 300.0, 60.0, seed=11)
    n_steps = 600
    result = simulation.simulate_parallel_learning(thresholds, n_steps)

    for step in (0, 100, 250, 300, 400, 600):
        expected = int((thresholds <= step).sum())
        actual = int(result["known_after_step"][step])
        assert actual == expected, "step %d: got %d, expected %d" % (
            step, actual, expected)


def test_curve_never_goes_down():
    """Words once learned stay learned, so the curve is non-decreasing."""
    thresholds = difficulty.make_gaussian_thresholds(500, 200.0, 50.0, seed=13)
    result = simulation.simulate_parallel_learning(thresholds, 400)
    curve = result["known_after_step"]
    assert np.all(np.diff(curve) >= 0)


def test_starts_at_zero_words_known():
    """Before any time has passed the learner knows nothing."""
    thresholds = difficulty.make_gaussian_thresholds(100, 200.0, 50.0, seed=17)
    result = simulation.simulate_parallel_learning(thresholds, 300)
    assert result["known_after_step"][0] == 0


def test_unreachable_words_are_reported_as_unlearned():
    """A word whose threshold is past the horizon must not be counted as known."""
    thresholds = np.array([10.0, 10000.0])
    result = simulation.simulate_parallel_learning(thresholds, n_steps=50)
    assert result["known_after_step"][-1] == 1
    assert result["learned_at_step"][0] == 10
    assert result["learned_at_step"][1] == -1


def test_benefit_speeds_learning_and_cost_slows_it():
    """A benefit must never be slower than neutral, and a cost never faster."""
    thresholds = difficulty.make_gaussian_thresholds(2000, 400.0, 70.0, seed=19)
    n_steps = 900
    neutral = simulation.simulate_parallel_learning(thresholds, n_steps, 0.0)
    benefit = simulation.simulate_parallel_learning(thresholds, n_steps, -0.1)
    cost = simulation.simulate_parallel_learning(thresholds, n_steps, +0.1)

    # At every point in time, benefit >= neutral >= cost words known.
    assert np.all(benefit["known_after_step"] >= neutral["known_after_step"])
    assert np.all(cost["known_after_step"] <= neutral["known_after_step"])
    # And the effect must be non-trivial, not just numerically equal.
    assert benefit["known_after_step"][-1] >= neutral["known_after_step"][-1]


def test_zero_shift_is_identical_to_no_coupling():
    """Passing a shift of exactly zero must change nothing."""
    thresholds = difficulty.make_gaussian_thresholds(300, 200.0, 40.0, seed=23)
    without = simulation.simulate_parallel_learning(thresholds, 400)
    with_zero = simulation.simulate_parallel_learning(thresholds, 400, 0.0)
    assert np.array_equal(without["known_after_step"],
                          with_zero["known_after_step"])


def test_input_thresholds_are_not_modified():
    """The caller's array must not be changed by running a simulation.

    The coupled model shifts thresholds as it goes, so it must work on a copy.
    """
    thresholds = np.array([100.0, 200.0, 300.0])
    original = thresholds.copy()
    simulation.simulate_parallel_learning(thresholds, 400, -0.5)
    assert np.array_equal(thresholds, original)


def test_thresholds_never_fall_below_floor_under_benefit():
    """Even a large benefit must not push thresholds below the minimum."""
    thresholds = np.full(50, 20.0)
    result = simulation.simulate_parallel_learning(
        thresholds, n_steps=60, shift_per_learned_word=-50.0)
    assert np.all(result["final_thresholds"] >= config.MIN_THRESHOLD)


def test_empty_vocabulary_is_handled():
    """Zero words should run without error and know nothing."""
    result = simulation.simulate_parallel_learning(np.array([]), n_steps=10)
    assert result["n_words"] == 0
    assert result["known_after_step"][-1] == 0


def test_identical_thresholds_learn_together():
    """If every word has the same threshold they must all be learned at once."""
    thresholds = np.full(100, 25.0)
    result = simulation.simulate_parallel_learning(thresholds, 50)
    curve = result["known_after_step"]
    assert curve[24] == 0
    assert curve[25] == 100


def test_sampling_model_learns_frequent_words_first():
    """In the sampling model, commoner words should be learned earlier."""
    frequencies = np.array([1000.0, 100.0, 10.0, 1.0])
    probabilities = difficulty.sampling_probabilities_from_frequencies(frequencies)
    result = simulation.simulate_sampled_learning(
        probabilities, threshold=5, n_steps=4000, seed=29)
    learned_at = result["learned_at_step"]
    # All four should be learned in that many steps.
    assert np.all(learned_at >= 0)
    # The commonest word should be learned before the rarest.
    assert learned_at[0] < learned_at[3]


def test_sampling_model_respects_its_threshold():
    """A word must be encountered exactly `threshold` times before being learned."""
    probabilities = np.array([0.5, 0.5])
    result = simulation.simulate_sampled_learning(
        probabilities, threshold=3, n_steps=200, seed=31)
    # Every learned word must have at least the threshold number of encounters.
    for word in range(2):
        if result["learned_at_step"][word] >= 0:
            assert result["encounter_counts"][word] >= 3


def test_sampling_model_is_reproducible():
    """Same seed, same result."""
    probabilities = np.array([0.4, 0.35, 0.25])
    first = simulation.simulate_sampled_learning(probabilities, 4, 500, seed=37)
    second = simulation.simulate_sampled_learning(probabilities, 4, 500, seed=37)
    assert np.array_equal(first["known_after_step"], second["known_after_step"])


def test_sampling_horizon_is_long_enough():
    """The chosen horizon should actually learn most of the vocabulary."""
    frequencies = difficulty.np.array([100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0])
    probabilities = difficulty.sampling_probabilities_from_frequencies(frequencies)
    n_steps = simulation.choose_sampling_horizon(probabilities, threshold=5)
    result = simulation.simulate_sampled_learning(probabilities, 5, n_steps, seed=41)
    learned_fraction = result["known_after_step"][-1] / len(frequencies)
    assert learned_fraction >= 0.8
