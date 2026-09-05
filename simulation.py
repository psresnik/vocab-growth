"""
The learning simulations.

Two models live here.

1. Parallel accrual (`simulate_parallel_learning`)
   Every word the learner has not yet learned gains one point per time step.
   When a word's points reach its threshold, it is learned. Optionally, each
   newly learned word nudges the thresholds of the remaining words up or down.

2. Fixed-threshold sampling (`simulate_sampled_learning`)
   Every word is equally hard, but only ONE word is encountered per time step,
   chosen at random with frequent words more likely. A word is learned once it
   has been encountered enough times.

Both return a dictionary with the same keys, so downstream analysis code does
not need to know which model produced the result.
"""

import numpy as np

import config


def simulate_parallel_learning(thresholds: np.ndarray,
                               n_steps: int,
                               shift_per_learned_word: float = 0.0,
                               min_threshold: float = config.MIN_THRESHOLD) -> dict:
    """Run the parallel-accrual model.

    On each time step, every not-yet-learned word gains exactly one point. Any
    word whose points have reached its threshold becomes learned.

    If `shift_per_learned_word` is not zero, then on each step the thresholds of
    all remaining unlearned words move by that amount for EACH word learned on
    that step. A negative value means learning a word helps with the others (a
    benefit); a positive value means it gets in the way (a cost).

    Args:
        thresholds: one threshold per word, in time steps.
        n_steps: how many time steps to run.
        shift_per_learned_word: points added to every unlearned word's threshold
            per word learned, on the step it is learned. Negative = benefit.
        min_threshold: thresholds are never allowed below this value.

    Returns:
        A dictionary with:
          "known_after_step"  : array of length n_steps + 1. Entry i is the
                                number of words known after i time steps, so
                                entry 0 is always 0.
          "learned_at_step"   : array with one entry per word, giving the step
                                at which it was learned, or -1 if it was never
                                learned within n_steps.
          "final_thresholds"  : the thresholds at the end of the run (these
                                differ from the input when a shift is applied).
          "n_words"           : how many words were simulated.
          "n_steps"           : how many steps were run.
    """
    thresholds = np.array(thresholds, dtype=float)  # copy: we may modify it
    n_words = len(thresholds)

    points = np.zeros(n_words)
    is_learned = np.zeros(n_words, dtype=bool)
    learned_at_step = np.full(n_words, -1)
    known_after_step = np.zeros(n_steps + 1, dtype=int)

    for step in range(1, n_steps + 1):
        # Every unlearned word gains a point. The `~is_learned` part selects
        # just those words; the loop-over-words version would be:
        #     for w in range(n_words):
        #         if not is_learned[w]: points[w] += 1
        points[~is_learned] += 1.0

        # A word is newly learned if it was not learned before and its points
        # have now reached its threshold.
        newly_learned = (~is_learned) & (points >= thresholds)
        n_new = int(newly_learned.sum())

        if n_new > 0:
            is_learned[newly_learned] = True
            learned_at_step[newly_learned] = step

            if shift_per_learned_word != 0.0:
                # Shift the thresholds of the words that are STILL unlearned.
                total_shift = shift_per_learned_word * n_new
                thresholds[~is_learned] += total_shift
                # Never let a threshold drop below the floor.
                thresholds = np.maximum(thresholds, min_threshold)

        known_after_step[step] = int(is_learned.sum())

    return {
        "known_after_step": known_after_step,
        "learned_at_step": learned_at_step,
        "final_thresholds": thresholds,
        "n_words": n_words,
        "n_steps": n_steps,
    }


def choose_sampling_horizon(probabilities: np.ndarray,
                            threshold: int,
                            coverage: float = 0.98,
                            max_steps: int = config.MAX_SAMPLING_STEPS) -> int:
    """Work out how many steps the sampling model needs to learn most words.

    A word encountered with probability p needs about threshold / p steps before
    it has been seen `threshold` times. To learn a given fraction of the
    vocabulary we need to wait for the rarest word in that fraction.

    Args:
        probabilities: probability of encountering each word on a step.
        threshold: how many encounters a word needs.
        coverage: fraction of words we want learned by the end, between 0 and 1.
        max_steps: hard cap, so the simulation always terminates.

    Returns:
        Number of time steps to run, as an integer.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    # Sort from most to least likely and pick the word at the coverage point.
    sorted_probabilities = np.sort(probabilities)[::-1]
    index = min(int(coverage * len(sorted_probabilities)), len(sorted_probabilities) - 1)
    slowest_probability = sorted_probabilities[index]
    estimate = int(1.3 * threshold / slowest_probability)
    return min(max(estimate, 100), max_steps)


def simulate_sampled_learning(probabilities: np.ndarray,
                              threshold: int,
                              n_steps: int,
                              seed: int) -> dict:
    """Run the fixed-threshold sampling model.

    Every word needs the same number of encounters, but on each time step only
    one word is encountered, drawn at random using `probabilities`. This shows
    that acceleration can come from how often words occur alone, without any
    differences in how hard the words are.

    Args:
        probabilities: probability of encountering each word on a step. Must sum
            to 1.
        threshold: how many encounters a word needs before it is learned.
        n_steps: how many time steps to run.
        seed: random seed.

    Returns:
        The same dictionary shape as simulate_parallel_learning, plus
        "encounter_counts": how many times each word was encountered.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    n_words = len(probabilities)
    rng = np.random.default_rng(seed)

    # Draw every encounter up front: this is much faster than drawing one at a
    # time, and the result is identical.
    encountered_word = rng.choice(n_words, size=n_steps, p=probabilities)

    encounter_counts = np.zeros(n_words, dtype=int)
    learned_at_step = np.full(n_words, -1)
    known_after_step = np.zeros(n_steps + 1, dtype=int)
    n_known = 0

    for step in range(1, n_steps + 1):
        word = encountered_word[step - 1]
        encounter_counts[word] += 1
        if encounter_counts[word] == threshold:
            learned_at_step[word] = step
            n_known += 1
        known_after_step[step] = n_known

    return {
        "known_after_step": known_after_step,
        "learned_at_step": learned_at_step,
        "encounter_counts": encounter_counts,
        "n_words": n_words,
        "n_steps": n_steps,
    }
