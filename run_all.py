"""
Run every experiment and write all figures and result tables.

Usage:
    python run_all.py                 # full run with the defaults in config.py
    python run_all.py --quick         # small, fast run for checking things work
    python run_all.py --seed 7        # different random seed
    python run_all.py --outdir myrun  # write somewhere else

Everything it writes goes into the output directory: five figures, a
results.csv summarising every run, and run_log.txt recording the exact settings
used.
"""

import argparse
import csv
import datetime
import os
import platform
import sys

import numpy as np

import analysis
import config
import datasets
import difficulty
import figures
import simulation


def parse_arguments(argv=None) -> argparse.Namespace:
    """Read command-line options.

    Args:
        argv: argument list, or None to read from the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="random seed (default: %(default)s)")
    parser.add_argument("--n-words", type=int, default=config.N_WORDS,
                        help="vocabulary size (default: %(default)s)")
    parser.add_argument("--n-steps", type=int, default=config.N_STEPS,
                        help="time steps to simulate (default: %(default)s)")
    parser.add_argument("--outdir", default=config.OUTPUT_DIR,
                        help="where to write figures (default: %(default)s)")
    parser.add_argument("--datadir", default=config.DATA_DIR,
                        help="where to look for data files (default: %(default)s)")
    parser.add_argument("--quick", action="store_true",
                        help="run a small fast version, for smoke-testing")
    return parser.parse_args(argv)


def run_baseline(n_words: int, n_steps: int, seed: int) -> dict:
    """Run the plain parallel-accrual model with Gaussian difficulty.

    Args:
        n_words: vocabulary size.
        n_steps: time steps to simulate.
        seed: random seed.

    Returns:
        Dictionary with "thresholds" and "result" (the simulation output).
    """
    thresholds = difficulty.make_gaussian_thresholds(
        n_words, config.THRESHOLD_MEAN, config.THRESHOLD_SD, seed)
    result = simulation.simulate_parallel_learning(thresholds, n_steps)
    return {"thresholds": thresholds, "result": result}


def run_benefit_and_cost(thresholds: np.ndarray, n_steps: int) -> dict:
    """Run the same vocabulary with a benefit and with a cost between words.

    Args:
        thresholds: the difficulty array to reuse, so the only thing that
            differs between conditions is the coupling.
        n_steps: time steps to simulate.

    Returns:
        Dictionary mapping condition name to growth curve.
    """
    neutral = simulation.simulate_parallel_learning(thresholds, n_steps, 0.0)
    benefit = simulation.simulate_parallel_learning(
        thresholds, n_steps, config.BENEFIT_SHIFT_PER_WORD)
    cost = simulation.simulate_parallel_learning(
        thresholds, n_steps, config.COST_SHIFT_PER_WORD)
    return {"neutral": neutral["known_after_step"],
            "benefit": benefit["known_after_step"],
            "cost": cost["known_after_step"]}


def run_frequency_models(n_words: int, data_dir: str) -> dict:
    """Build difficulty from word frequencies and run both registers.

    Args:
        n_words: how many words to take from each frequency list.
        data_dir: where to look for real frequency data.

    Returns:
        Dictionary with a growth curve and a source label per register.
    """
    output = {}
    for register in ("child", "adult"):
        frequency_data = datasets.get_frequencies(register, data_dir, n_words)
        thresholds = difficulty.thresholds_from_frequencies(
            frequency_data["frequencies"], config.FREQ_BASE, config.FREQ_SCALE)
        # Run long enough for the hardest word to be reachable.
        n_steps = int(thresholds.max() * 1.05) + 10
        result = simulation.simulate_parallel_learning(thresholds, n_steps)
        output[register] = {"curve": result["known_after_step"],
                            "source": frequency_data["source"],
                            "is_real": frequency_data["is_real"],
                            "thresholds": thresholds}
    return output


def run_sampling_model(n_words: int, data_dir: str, seed: int,
                       threshold: int) -> dict:
    """Run the fixed-threshold sampling model on child-directed frequencies.

    Args:
        n_words: how many words to simulate.
        data_dir: where to look for real frequency data.
        seed: random seed.
        threshold: encounters needed per word.

    Returns:
        Dictionary with "curve" and "source".
    """
    frequency_data = datasets.get_frequencies("child", data_dir, n_words)
    probabilities = difficulty.sampling_probabilities_from_frequencies(
        frequency_data["frequencies"])
    n_steps = simulation.choose_sampling_horizon(probabilities, threshold)
    result = simulation.simulate_sampled_learning(
        probabilities, threshold, n_steps, seed)
    return {"curve": result["known_after_step"],
            "source": frequency_data["source"]}


def run_distribution_comparison(n_words: int, n_steps: int, seed: int) -> dict:
    """Run the same model under every difficulty distribution in the registry.

    Args:
        n_words: vocabulary size.
        n_steps: time steps to simulate.
        seed: random seed.

    Returns:
        Dictionary mapping distribution name to growth curve.
    """
    curves = {}
    for name in sorted(difficulty.DIFFICULTY_MAKERS):
        thresholds = difficulty.make_thresholds(
            name, n_words, config.THRESHOLD_MEAN, config.THRESHOLD_SD, seed)
        # The exponential distribution has a very long tail, so give every
        # condition enough time to finish rather than truncating one of them.
        steps_needed = min(int(np.percentile(thresholds, 99)) + 10, n_steps * 4)
        result = simulation.simulate_parallel_learning(
            thresholds, max(steps_needed, n_steps))
        curves[name] = result["known_after_step"]
    return curves


def write_results_table(rows: list, path: str) -> None:
    """Write the summary table of every run to a CSV file.

    Args:
        rows: list of dictionaries, all with the same keys.
        path: where to write the file.
    """
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_run_log(arguments: argparse.Namespace, notes: list, path: str) -> None:
    """Record exactly how this run was configured, for reproducibility.

    Args:
        arguments: the parsed command-line arguments.
        notes: extra lines to include, such as which data sources were used.
        path: where to write the file.
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Vocabulary growth model - run log\n")
        handle.write("timestamp: %s\n" % datetime.datetime.now().isoformat())
        handle.write("python: %s\n" % sys.version.split()[0])
        handle.write("platform: %s\n" % platform.platform())
        handle.write("numpy: %s\n" % np.__version__)
        handle.write("\nsettings:\n")
        for key, value in sorted(vars(arguments).items()):
            handle.write("  %s = %s\n" % (key, value))
        handle.write("\nconstants from config.py:\n")
        for key in sorted(dir(config)):
            if key.isupper():
                handle.write("  %s = %s\n" % (key, getattr(config, key)))
        handle.write("\nnotes:\n")
        for note in notes:
            handle.write("  %s\n" % note)


def main(argv=None) -> int:
    """Run everything and write all outputs.

    Args:
        argv: argument list, or None to read from the command line.

    Returns:
        Process exit code: 0 on success.
    """
    arguments = parse_arguments(argv)

    if arguments.quick:
        arguments.n_words = 800
        arguments.n_steps = 6000

    os.makedirs(arguments.outdir, exist_ok=True)
    notes = []
    summary_rows = []

    print("Vocabulary growth model")
    print("  words: %d   steps: %d   seed: %d"
          % (arguments.n_words, arguments.n_steps, arguments.seed))

    # --- 1. Baseline -------------------------------------------------------
    baseline = run_baseline(arguments.n_words, arguments.n_steps, arguments.seed)
    baseline_curve = baseline["result"]["known_after_step"]
    baseline_summary = analysis.describe_growth(baseline_curve, config.RATE_WINDOW)

    early_window = (0, int(config.THRESHOLD_MEAN - 2 * config.THRESHOLD_SD))
    later_window = (early_window[1], early_window[1] + 600)
    figures.plot_difficulty_distribution(
        baseline["thresholds"], [early_window, later_window],
        os.path.join(arguments.outdir, "fig1_difficulty_distribution.png"))
    figures.plot_rate_and_curve(
        baseline_curve, config.RATE_WINDOW,
        os.path.join(arguments.outdir, "fig2_baseline_growth_and_rate.png"),
        "Baseline: constant effort, accelerating vocabulary")

    early_count = analysis.words_learned_in_window(baseline_curve, *early_window)
    later_count = analysis.words_learned_in_window(baseline_curve, *later_window)
    print("\n1. Baseline")
    print("   words learned in steps %d-%d: %d" % (early_window + (early_count,)))
    print("   words learned in steps %d-%d: %d" % (later_window + (later_count,)))
    print("   the later window is %.1f times denser, over a window %.1fx as wide"
          % (later_count / max(early_count, 1),
             (later_window[1] - later_window[0]) /
             max(early_window[1] - early_window[0], 1)))
    print("   accelerates: %s   decelerates at end: %s"
          % (baseline_summary["accelerates"],
             baseline_summary["decelerates_at_end"]))
    summary_rows.append(_summary_row("baseline", baseline_summary))

    # --- 2. Benefit and cost ----------------------------------------------
    coupled = run_benefit_and_cost(baseline["thresholds"], arguments.n_steps)
    figures.plot_growth_curves(
        coupled,
        os.path.join(arguments.outdir, "fig3_benefit_and_cost.png"),
        "Acceleration survives both help and hindrance")
    print("\n2. Benefit and cost")
    halfway = arguments.n_words // 2
    for name, curve in coupled.items():
        condition_summary = analysis.describe_growth(curve, config.RATE_WINDOW)
        print("   %-8s reaches %d words at step %5d   accelerates: %s"
              % (name, halfway, analysis.step_reaching(curve, halfway),
                 condition_summary["accelerates"]))
        summary_rows.append(_summary_row(name, condition_summary))

    # --- 3. Frequency-based difficulty ------------------------------------
    frequency_runs = run_frequency_models(config.N_FREQUENT_WORDS,
                                          arguments.datadir)
    figures.plot_growth_curves(
        {name: run["curve"] for name, run in frequency_runs.items()},
        os.path.join(arguments.outdir, "fig4_frequency_models.png"),
        "Difficulty built from word frequency")
    print("\n3. Frequency-based difficulty")
    for register, run in frequency_runs.items():
        run_summary = analysis.describe_growth(run["curve"], config.RATE_WINDOW)
        print("   %-6s source: %s" % (register, run["source"]))
        print("          accelerates: %s   final vocabulary: %d"
              % (run_summary["accelerates"], run_summary["final_known"]))
        summary_rows.append(_summary_row("frequency_" + register, run_summary))
        notes.append("frequency source (%s): %s" % (register, run["source"]))

    # --- 4. Fixed-threshold sampling --------------------------------------
    sampling = run_sampling_model(config.N_FREQUENT_WORDS, arguments.datadir,
                                  arguments.seed, config.SAMPLING_THRESHOLD)
    figures.plot_rate_and_curve(
        sampling["curve"], max(config.RATE_WINDOW * 10, 1),
        os.path.join(arguments.outdir, "fig5_sampling_model.png"),
        "Every word equally hard: only how often you hear it varies")
    sampling_summary = analysis.describe_growth(sampling["curve"],
                                                config.RATE_WINDOW * 10)
    print("\n4. Fixed-threshold sampling model")
    print("   all words equally difficult, %d encounters needed each"
          % config.SAMPLING_THRESHOLD)
    print("   accelerates: %s   final vocabulary: %d"
          % (sampling_summary["accelerates"], sampling_summary["final_known"]))
    summary_rows.append(_summary_row("sampling", sampling_summary))

    # --- 5. Distribution comparison ---------------------------------------
    comparison = run_distribution_comparison(
        min(arguments.n_words, 4000), arguments.n_steps, arguments.seed)
    figures.plot_distribution_comparison(
        comparison, config.RATE_WINDOW,
        os.path.join(arguments.outdir, "fig6_distribution_comparison.png"))
    print("\n5. Which difficulty distributions produce acceleration?")
    for name, curve in comparison.items():
        distribution_summary = analysis.describe_growth(curve, config.RATE_WINDOW)
        print("   %-12s accelerates: %-5s  peak/early rate: %6.2f"
              % (name, distribution_summary["accelerates"],
                 distribution_summary["acceleration_ratio"]))
        summary_rows.append(_summary_row("dist_" + name, distribution_summary))

    # --- 6. Comparison with real norms, if available -----------------------
    norms = datasets.get_wordbank_norms(arguments.datadir)
    if norms is None:
        print("\n6. Wordbank norms not found in %s, skipping that comparison."
              % arguments.datadir)
        notes.append("wordbank norms: not available, comparison figure skipped")
    else:
        figures.plot_model_against_norms(
            baseline_curve, norms,
            os.path.join(arguments.outdir, "fig7_model_vs_norms.png"))
        print("\n6. Compared model against Wordbank norms for ages %d-%d months."
              % (norms["ages"].min(), norms["ages"].max()))
        notes.append("wordbank norms: loaded, %d ages" % len(norms["ages"]))

    write_results_table(summary_rows,
                        os.path.join(arguments.outdir, "results.csv"))
    write_run_log(arguments, notes,
                  os.path.join(arguments.outdir, "run_log.txt"))

    print("\nWrote figures, results.csv and run_log.txt to %s/" % arguments.outdir)
    return 0


def _summary_row(name: str, summary: dict) -> dict:
    """Flatten one growth summary into a row for the results table.

    Args:
        name: label for this run.
        summary: dictionary from analysis.describe_growth.

    Returns:
        Dictionary suitable for a CSV row.
    """
    quarters = summary["quarters"]
    return {
        "run": name,
        "accelerates": summary["accelerates"],
        "acceleration_ratio": round(summary["acceleration_ratio"], 3),
        "early_rate": round(summary["early_rate"], 2),
        "peak_rate": round(summary["peak_rate"], 2),
        "peak_step": summary["peak_step"],
        "decelerates_at_end": summary["decelerates_at_end"],
        "final_known": summary["final_known"],
        "span_start": summary["span_start"],
        "span_end": summary["span_end"],
        "quarter1": quarters[0],
        "quarter2": quarters[1],
        "quarter3": quarters[2],
        "quarter4": quarters[3],
    }


if __name__ == "__main__":
    sys.exit(main())
