"""
Drawing the figures.

Each function here takes already-computed results and writes one image file.
None of them run simulations themselves, so you can call them on results you
produced any way you like.

All figures use the "Agg" backend, which draws straight to a file and never
needs a screen. That makes the whole project runnable over ssh or in a script.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import analysis


def _finish(figure, path: str) -> str:
    """Save a figure and close it, so long runs do not leak memory.

    Args:
        figure: the matplotlib figure to save.
        path: where to write the image.

    Returns:
        The path written, for convenience when logging.
    """
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_difficulty_distribution(thresholds: np.ndarray,
                                 window_starts: list,
                                 path: str) -> str:
    """Draw the histogram of word difficulties, with two windows highlighted.

    The point of the highlighting is to show that a LATER window of the same
    width contains more words than an EARLIER one, which is the whole reason
    growth accelerates.

    Args:
        thresholds: one threshold per word, in time steps.
        window_starts: list of two (start, end) tuples to shade.
        path: where to write the image.

    Returns:
        The path written.
    """
    figure, axis = plt.subplots(figsize=(6, 4))
    counts, edges, _ = axis.hist(thresholds, bins=120, color="0.25")

    colours = ["tab:blue", "tab:red"]
    for index, (start, end) in enumerate(window_starts):
        in_window = (edges[:-1] >= start) & (edges[:-1] < end)
        axis.bar(edges[:-1][in_window], counts[in_window],
                 width=np.diff(edges)[0], color=colours[index], align="edge",
                 label="steps %d-%d: %d words"
                       % (start, end, int(((thresholds >= start) &
                                           (thresholds < end)).sum())))

    axis.set_xlabel("Time to acquisition (time steps)\n<- easier            harder ->")
    axis.set_ylabel("Number of words")
    axis.set_title("How word difficulty is distributed")
    axis.legend(fontsize=8)
    return _finish(figure, path)


def plot_growth_curves(curves: dict, path: str, title: str,
                       xlabel: str = "Time steps",
                       ylabel: str = "Words known") -> str:
    """Draw one or more growth curves on the same axes.

    Args:
        curves: mapping from label to growth-curve array.
        path: where to write the image.
        title: figure title.
        xlabel, ylabel: axis labels.

    Returns:
        The path written.
    """
    figure, axis = plt.subplots(figsize=(6, 4))
    for label, curve in curves.items():
        axis.plot(np.arange(len(curve)), curve, label=label, linewidth=1.8)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if len(curves) > 1:
        axis.legend(fontsize=9)
    return _finish(figure, path)


def plot_rate_and_curve(growth_curve: np.ndarray, window: int,
                        path: str, title: str) -> str:
    """Draw a growth curve next to its learning rate.

    Seeing the two together makes the acceleration-then-deceleration pattern
    obvious: the rate curve rises to a peak and then falls, even though the
    learner never changes how fast it accumulates evidence.

    Args:
        growth_curve: number of words known after each step.
        window: width of the rate-measuring window, in time steps.
        path: where to write the image.
        title: figure title.

    Returns:
        The path written.
    """
    rates = analysis.learning_rate(growth_curve, window)
    rate_steps = (np.arange(len(rates)) + 0.5) * window

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    top.plot(np.arange(len(growth_curve)), growth_curve, color="black")
    top.set_ylabel("Words known")
    top.set_title(title)

    bottom.plot(rate_steps, rates, color="tab:purple")
    bottom.set_ylabel("Words learned\nper %d steps" % window)
    bottom.set_xlabel("Time steps")
    bottom.axvline(analysis.peak_rate_step(growth_curve, window),
                   color="0.5", linestyle="--", linewidth=1,
                   label="fastest learning")
    bottom.legend(fontsize=8)
    return _finish(figure, path)


def plot_distribution_comparison(results: dict, window: int, path: str) -> str:
    """Draw one small panel per difficulty distribution.

    Each panel shows the growth curve (black, left axis) and the learning rate
    (purple, right axis) for one difficulty distribution. Small separate panels
    work better than one crowded plot here, because several of the
    distributions produce almost identical curves -- which is itself the point.

    Args:
        results: mapping from distribution name to growth-curve array.
        window: rate window used for the acceleration test.
        path: where to write the image.

    Returns:
        The path written.
    """
    names = list(results)
    n_panels = len(names)
    n_columns = 4
    n_rows = int(np.ceil(n_panels / n_columns))

    figure, axes = plt.subplots(n_rows, n_columns,
                                figsize=(3.1 * n_columns, 2.6 * n_rows))
    flat_axes = np.atleast_1d(axes).ravel()

    for panel_index, name in enumerate(names):
        curve = results[name]
        axis = flat_axes[panel_index]
        summary = analysis.describe_growth(curve, window)

        axis.plot(np.arange(len(curve)), curve / max(curve[-1], 1),
                  color="black", linewidth=1.5)
        axis.set_ylim(-0.05, 1.05)

        rates = analysis.learning_rate(curve, window)
        rate_steps = (np.arange(len(rates)) + 0.5) * window
        rate_axis = axis.twinx()
        rate_axis.plot(rate_steps, rates / max(rates.max(), 1),
                       color="tab:purple", linewidth=1.0, alpha=0.75)
        rate_axis.set_ylim(-0.05, 1.15)
        rate_axis.set_yticks([])

        verdict = "accelerates" if summary["accelerates"] else "no acceleration"
        axis.set_title("%s\n%s (peak/start = %.1fx)"
                       % (name, verdict, summary["acceleration_ratio"]),
                       fontsize=9)
        axis.tick_params(labelsize=7)

    # Hide any unused panels in the grid.
    for empty_index in range(n_panels, len(flat_axes)):
        flat_axes[empty_index].axis("off")

    figure.suptitle("Black: vocabulary known.   Purple: learning rate.",
                    fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_model_against_norms(growth_curve: np.ndarray,
                             norms: dict,
                             path: str) -> str:
    """Compare the model's growth curve against real child vocabulary norms.

    The two are plotted on separate x axes, because model time steps are not
    calibrated to months. What is being compared is the SHAPE of the two curves,
    not their absolute values.

    Args:
        growth_curve: number of words known after each step.
        norms: dictionary from datasets.load_wordbank_norms.
        path: where to write the image.

    Returns:
        The path written.
    """
    figure, axis = plt.subplots(figsize=(6, 4))

    model_fraction = growth_curve / max(growth_curve[-1], 1)
    model_time = np.linspace(0, 1, len(growth_curve))
    axis.plot(model_time, model_fraction, color="black", label="model")

    norm_fraction = (norms["mean_words_produced"] /
                     max(norms["mean_words_produced"].max(), 1))
    norm_time = ((norms["ages"] - norms["ages"].min()) /
                 max(norms["ages"].max() - norms["ages"].min(), 1))
    axis.plot(norm_time, norm_fraction, "o-", color="tab:green",
              label="Wordbank CDI norms")

    axis.set_xlabel("Rescaled time (model steps / child age, both 0 to 1)")
    axis.set_ylabel("Fraction of final vocabulary")
    axis.set_title("Model shape vs. real vocabulary growth")
    axis.legend(fontsize=9)
    return _finish(figure, path)
