"""
Worked example: adding your own difficulty distribution.

Run it with:
    python experiments/example_new_distribution.py

The point of this example is that you do NOT have to edit any of the simulation
code to try out a new idea. You write one function and add one line to the
registry, and everything else -- the simulation, the analysis, the figures --
works unchanged.

The distribution invented here is a "two-speed vocabulary": most words are
moderately hard, but a small group of words is much easier than the rest,
perhaps because a child hears them constantly. The question is whether that
head start changes the overall shape of growth.
"""

import os
import sys

import numpy as np

# Make the project modules importable when running from the experiments folder.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import analysis
import config
import difficulty
import figures
import simulation


def make_head_start_thresholds(n_words: int, mean: float, sd: float,
                               seed: int) -> np.ndarray:
    """Most words are ordinary, but one word in ten is much easier.

    Note the signature: it takes exactly the same arguments as every other
    generator in difficulty.py. That is what lets the registry treat them
    interchangeably.

    Args:
        n_words: how many words to generate thresholds for.
        mean: average threshold for the ordinary words, in time steps.
        sd: spread of the ordinary words, in time steps.
        seed: random seed.

    Returns:
        Array of n_words thresholds, in time steps.
    """
    rng = np.random.default_rng(seed)
    thresholds = rng.normal(mean, sd, size=n_words)

    # Pick a tenth of the words at random and make them much easier.
    n_easy = n_words // 10
    easy_words = rng.choice(n_words, size=n_easy, replace=False)
    thresholds[easy_words] = rng.normal(mean * 0.4, sd * 0.5, size=n_easy)

    return np.maximum(thresholds, config.MIN_THRESHOLD)


def main() -> int:
    """Register the new distribution, run it, and compare it to the standard one."""
    # This single line makes the new distribution available everywhere.
    difficulty.DIFFICULTY_MAKERS["head_start"] = make_head_start_thresholds

    n_words = 6000
    n_steps = 7000
    curves = {}

    for name in ("gaussian", "head_start"):
        thresholds = difficulty.make_thresholds(
            name, n_words, config.THRESHOLD_MEAN, config.THRESHOLD_SD,
            seed=config.SEED)
        result = simulation.simulate_parallel_learning(thresholds, n_steps)
        curve = result["known_after_step"]
        curves[name] = curve

        summary = analysis.describe_growth(curve, config.RATE_WINDOW)
        print("%-12s accelerates: %-5s  peak/start rate: %5.2f  peak at step %d"
              % (name, summary["accelerates"], summary["acceleration_ratio"],
                 summary["peak_step"]))

    output_path = os.path.join(config.OUTPUT_DIR, "experiment_head_start.png")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    figures.plot_growth_curves(
        curves, output_path,
        "Does an easy head-start group change the shape of growth?")
    print("\nWrote %s" % output_path)

    print("\nWhat to notice: the head-start group produces an early bump, but "
          "the main spurt still happens at the same place and for the same "
          "reason. Try changing the size of the easy group and see how big it "
          "has to get before the shape really changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
