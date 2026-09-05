"""
Worked example: sweeping a parameter to see what it controls.

Run it with:
    python experiments/example_parameter_sweep.py

This asks a specific question: how does the SPREAD of word difficulty change
the vocabulary spurt? The model says acceleration comes from variation in how
hard words are, so more variation should mean a more dramatic spurt. Does it?

The pattern shown here -- loop over parameter values, collect a summary number
for each, plot the summaries -- is the one you will use for most experiments.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import analysis
import config
import difficulty
import simulation


def measure_one_spread(spread: float, n_words: int, seed: int) -> dict:
    """Run the model once with a given difficulty spread and summarise it.

    Args:
        spread: standard deviation of the difficulty distribution, in time steps.
        n_words: vocabulary size.
        seed: random seed.

    Returns:
        The dictionary from analysis.describe_growth.
    """
    thresholds = difficulty.make_gaussian_thresholds(
        n_words, config.THRESHOLD_MEAN, spread, seed)
    # Run long enough that nearly every word gets learned whatever the spread.
    n_steps = int(config.THRESHOLD_MEAN + 5 * spread) + 100
    result = simulation.simulate_parallel_learning(thresholds, n_steps)
    return analysis.describe_growth(result["known_after_step"],
                                    config.RATE_WINDOW)


def main() -> int:
    """Sweep the difficulty spread and plot what it does to the spurt."""
    spreads = [200.0, 400.0, 680.0, 1000.0, 1500.0, 2000.0]
    n_words = 5000

    ratios = []
    peak_rates = []

    print("%-10s %-12s %-14s %s" % ("spread", "accelerates", "peak/start", "peak rate"))
    for spread in spreads:
        summary = measure_one_spread(spread, n_words, config.SEED)
        ratios.append(summary["acceleration_ratio"])
        peak_rates.append(summary["peak_rate"])
        print("%-10.0f %-12s %-14.2f %.1f"
              % (spread, summary["accelerates"], summary["acceleration_ratio"],
                 summary["peak_rate"]))

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(spreads, ratios, "o-", color="tab:blue")
    left.set_xlabel("Spread of word difficulty (time steps)")
    left.set_ylabel("Peak rate / starting rate")
    left.set_title("How pronounced is the spurt?")

    right.plot(spreads, peak_rates, "o-", color="tab:purple")
    right.set_xlabel("Spread of word difficulty (time steps)")
    right.set_ylabel("Words learned per %d steps at the peak" % config.RATE_WINDOW)
    right.set_title("How fast is the fastest moment?")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_DIR, "experiment_spread_sweep.png")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print("\nWrote %s" % output_path)

    print("\nWhat to notice: a narrow spread means all the words arrive at once, "
          "which is a very sharp spurt but a short one. A wide spread means "
          "learning is spread out. Think about which of these looks more like a "
          "real child, and what that implies about how varied real words are.")
    print("\nAlso notice the largest spread breaking the pattern. Once the "
          "spread gets close to the mean, the bell curve runs off the left-hand "
          "end into negative difficulty, and those words get clipped to the "
          "minimum threshold instead. That creates a clump of words learned "
          "immediately, which inflates the starting rate. This is a modelling "
          "artefact, not a finding: a good habit is to plot the difficulty "
          "distribution itself whenever a result looks surprising.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
