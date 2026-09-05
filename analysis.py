"""
Measuring the shape of a growth curve.

The input to almost everything here is a "growth curve": an array where entry i
is the number of words known after i time steps. That is exactly what the
simulation functions return under the key "known_after_step".

The questions we want to answer:
  - How fast is the learner picking up words at each moment?  -> learning_rate
  - Is that rate rising or falling?                           -> rate_change
  - When is learning fastest?                                 -> peak_rate_step
  - Does this run accelerate at all?                          -> describe_growth
"""

import numpy as np


def words_learned_in_window(growth_curve: np.ndarray,
                            start_step: int,
                            end_step: int) -> int:
    """Count how many words were learned between two time points.

    Args:
        growth_curve: number of words known after each step.
        start_step: first step of the window (inclusive of the state at that step).
        end_step: last step of the window.

    Returns:
        Number of words learned in the interval, as an integer.

    Raises:
        ValueError: if the window is backwards or runs past the end of the curve.
    """
    if end_step < start_step:
        raise ValueError("end_step must not be before start_step.")
    if end_step >= len(growth_curve):
        raise ValueError("end_step %d is past the end of the curve (length %d)."
                         % (end_step, len(growth_curve)))
    return int(growth_curve[end_step] - growth_curve[start_step])


def learning_rate(growth_curve: np.ndarray, window: int) -> np.ndarray:
    """Words learned per window, measured across the whole run.

    This is the first derivative of the growth curve: how fast vocabulary is
    growing at each point in time.

    Args:
        growth_curve: number of words known after each step.
        window: width of the measuring window, in time steps.

    Returns:
        Array of counts. Entry i is the number of words learned between step
        i * window and step (i + 1) * window.
    """
    if window < 1:
        raise ValueError("window must be at least 1 time step.")
    # Take the curve's value at each window boundary, then difference it.
    boundaries = growth_curve[::window]
    return np.diff(boundaries)


def rate_change(growth_curve: np.ndarray, window: int) -> np.ndarray:
    """Change in the learning rate from one window to the next.

    This is the second derivative. Positive values mean the learner is speeding
    up (acceleration); negative values mean slowing down (deceleration).

    Args:
        growth_curve: number of words known after each step.
        window: width of the measuring window, in time steps.

    Returns:
        Array of changes, one shorter than the learning_rate array.
    """
    return np.diff(learning_rate(growth_curve, window))


def peak_rate_step(growth_curve: np.ndarray, window: int) -> int:
    """Find the time step at which learning is fastest.

    Args:
        growth_curve: number of words known after each step.
        window: width of the measuring window, in time steps.

    Returns:
        The time step at the centre of the fastest window.
    """
    rates = learning_rate(growth_curve, window)
    fastest_window = int(np.argmax(rates))
    return int((fastest_window + 0.5) * window)


def step_reaching(growth_curve: np.ndarray, n_words_known: int) -> int:
    """Find the first step at which the learner knows at least this many words.

    Useful for comparing runs: "the benefit condition reached 3000 words 400
    steps earlier than the baseline".

    Args:
        growth_curve: number of words known after each step.
        n_words_known: the vocabulary size to look for.

    Returns:
        The first step reaching that size, or -1 if it is never reached.
    """
    reached = np.where(growth_curve >= n_words_known)[0]
    if len(reached) == 0:
        return -1
    return int(reached[0])


def active_span(growth_curve: np.ndarray,
                low_fraction: float = 0.01,
                high_fraction: float = 0.99) -> tuple:
    """Find the period during which most of the learning actually happens.

    Growth curves often have very long flat tails: one unlucky word might not
    be learned until ten times later than all the others. Measuring "the first
    half of learning" across the whole run would then be misleading. So we trim
    to the period between the first 1% and the last 1% of the vocabulary.

    Args:
        growth_curve: number of words known after each step.
        low_fraction: fraction of the final vocabulary marking the start.
        high_fraction: fraction marking the end.

    Returns:
        Tuple (start_step, end_step). If nothing was ever learned, returns
        (0, len(growth_curve) - 1).
    """
    final_known = int(growth_curve[-1])
    if final_known <= 0:
        return 0, len(growth_curve) - 1

    start = step_reaching(growth_curve, max(int(low_fraction * final_known), 1))
    end = step_reaching(growth_curve, int(high_fraction * final_known))
    if start < 0:
        start = 0
    if end < 0 or end <= start:
        end = len(growth_curve) - 1
    return start, end


def describe_growth(growth_curve: np.ndarray, window: int,
                    acceleration_threshold: float = 1.5) -> dict:
    """Summarise the shape of one growth curve.

    A run counts as accelerating if the learner gets substantially FASTER after
    learning gets going: the fastest learning rate must be at least
    `acceleration_threshold` times the rate at the start, and it must happen
    later than the start.

    Measuring it this way, rather than comparing halves or quarters of the run,
    matters because some curves accelerate briefly and then spend a long time
    crawling through their hardest words. Comparing the early rate to the peak
    rate catches those; comparing the first half to the second does not.

    Args:
        growth_curve: number of words known after each step.
        window: width of the measuring window, in time steps.
        acceleration_threshold: how many times faster the peak has to be before
            we call it acceleration.

    Returns:
        Dictionary with:
          "accelerates"        : True if the run speeds up as described above.
          "acceleration_ratio" : peak rate divided by the starting rate.
          "early_rate"         : words per window at the start of learning.
          "peak_rate"          : words per window at the fastest moment.
          "peak_step"          : step at which learning was fastest.
          "decelerates_at_end" : True if the run ends slower than its peak.
          "final_known"        : words known at the end of the run.
          "span_start", "span_end" : the trimmed learning period used.
          "quarters"           : words learned in each quarter of that period,
                                 reported for interest but not used above.
    """
    final_known = int(growth_curve[-1])
    start, end = active_span(growth_curve)

    # Words learned in each quarter of the active period, for reporting.
    quarter_width = max((end - start) // 4, 1)
    quarters = []
    for index in range(4):
        window_start = start + index * quarter_width
        window_end = min(start + (index + 1) * quarter_width,
                         len(growth_curve) - 1)
        quarters.append(words_learned_in_window(growth_curve,
                                                window_start, window_end))

    segment = growth_curve[start:end + 1]
    rates = learning_rate(segment, window)

    if len(rates) < 4 or final_known == 0:
        # Too short to say anything meaningful about shape.
        return {
            "accelerates": False,
            "acceleration_ratio": 0.0,
            "early_rate": 0.0,
            "peak_rate": 0.0,
            "peak_step": start,
            "decelerates_at_end": False,
            "final_known": final_known,
            "span_start": start,
            "span_end": end,
            "quarters": quarters,
        }

    # "Early" is the first tenth of the learning period, averaged so that one
    # noisy window cannot decide the answer.
    n_early = max(len(rates) // 10, 1)
    early_rate = float(rates[:n_early].mean())
    peak_rate = float(rates.max())
    peak_index = int(np.argmax(rates))
    late_rate = float(rates[-n_early:].mean())

    if early_rate > 0:
        ratio = peak_rate / early_rate
    else:
        ratio = float("inf") if peak_rate > 0 else 0.0

    accelerates = (ratio >= acceleration_threshold) and (peak_index >= n_early)

    return {
        "accelerates": accelerates,
        "acceleration_ratio": ratio,
        "early_rate": early_rate,
        "peak_rate": peak_rate,
        "peak_step": start + int((peak_index + 0.5) * window),
        "decelerates_at_end": late_rate < peak_rate,
        "final_known": final_known,
        "span_start": start,
        "span_end": end,
        "quarters": quarters,
    }
