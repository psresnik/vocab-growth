"""
Building time-to-acquisition thresholds.

A "threshold" is the number of points a word needs before the learner knows it.
A whole vocabulary is therefore described by one array of thresholds, one entry
per word. Everything in this file returns such an array.

The central idea of the model is that the SHAPE of this array determines the
shape of the growth curve. Words are learned in parallel at a constant rate, so
the number of words known at time t is simply the number of thresholds that are
less than or equal to t. If there are few easy words and many moderate ones,
growth accelerates.
"""

import numpy as np

import config


# ---------------------------------------------------------------------------
# Distribution-based thresholds
# ---------------------------------------------------------------------------

def make_gaussian_thresholds(n_words: int, mean: float, sd: float,
                             seed: int) -> np.ndarray:
    """Draw thresholds from a normal (bell-shaped) distribution.

    This is the default difficulty distribution: a few easy words, many
    moderate ones, a few very hard ones.

    Args:
        n_words: how many words to generate thresholds for.
        mean: average threshold, in time steps.
        sd: standard deviation of the thresholds, in time steps.
        seed: random seed, so the same call always gives the same array.

    Returns:
        Array of n_words thresholds, in time steps. Values are clipped so none
        falls below config.MIN_THRESHOLD.
    """
    rng = np.random.default_rng(seed)
    thresholds = rng.normal(loc=mean, scale=sd, size=n_words)
    # np.maximum compares every element against the floor value at once; the
    # loop-over-words version would be: [max(t, MIN) for t in thresholds]
    return np.maximum(thresholds, config.MIN_THRESHOLD)


def make_uniform_thresholds(n_words: int, mean: float, sd: float,
                            seed: int) -> np.ndarray:
    """Draw thresholds evenly across a range, so every difficulty is equally common.

    Included as a comparison case: with a flat distribution the growth curve is
    a straight line, neither accelerating nor decelerating.

    Args (same meaning as make_gaussian_thresholds). The range is chosen to have
    the requested mean and standard deviation.
    """
    rng = np.random.default_rng(seed)
    # A uniform distribution on [a, b] has sd = (b - a) / sqrt(12).
    half_width = sd * np.sqrt(3.0)
    thresholds = rng.uniform(mean - half_width, mean + half_width, size=n_words)
    return np.maximum(thresholds, config.MIN_THRESHOLD)


def make_exponential_thresholds(n_words: int, mean: float, sd: float,
                                seed: int) -> np.ndarray:
    """Draw thresholds from an exponential distribution: many easy words, few hard ones.

    This is the important negative case. Here difficulty is INVERSELY related to
    how many words sit at that difficulty, which is the opposite of the pattern
    that produces acceleration. Growth should decelerate from the very start.

    The `sd` argument is ignored (an exponential distribution's spread is fixed
    by its mean); it is accepted so that every generator has the same signature.
    """
    rng = np.random.default_rng(seed)
    thresholds = rng.exponential(scale=mean, size=n_words)
    return np.maximum(thresholds, config.MIN_THRESHOLD)


def make_lognormal_thresholds(n_words: int, mean: float, sd: float,
                              seed: int) -> np.ndarray:
    """Draw thresholds from a log-normal distribution: skewed, with a long hard tail.

    Args (same meaning as make_gaussian_thresholds). The underlying normal is
    chosen so the resulting values have roughly the requested mean and sd.
    """
    rng = np.random.default_rng(seed)
    # Standard formulas converting a target mean/sd into log-space parameters.
    variance_ratio = 1.0 + (sd / mean) ** 2
    log_sd = np.sqrt(np.log(variance_ratio))
    log_mean = np.log(mean) - 0.5 * log_sd ** 2
    thresholds = rng.lognormal(mean=log_mean, sigma=log_sd, size=n_words)
    return np.maximum(thresholds, config.MIN_THRESHOLD)


def make_gamma_thresholds(n_words: int, mean: float, sd: float,
                          seed: int) -> np.ndarray:
    """Draw thresholds from a gamma distribution.

    With a large shape parameter this looks almost Gaussian; with a small one it
    looks almost exponential. Useful for showing that the family a distribution
    belongs to matters less than its shape.
    """
    rng = np.random.default_rng(seed)
    shape = (mean / sd) ** 2
    scale = sd ** 2 / mean
    thresholds = rng.gamma(shape=shape, scale=scale, size=n_words)
    return np.maximum(thresholds, config.MIN_THRESHOLD)


def make_bimodal_thresholds(n_words: int, mean: float, sd: float,
                            seed: int) -> np.ndarray:
    """Draw thresholds from two separate bell curves, one easy group and one hard group.

    Produces a growth curve with TWO spurts rather than one, which is a good
    demonstration that the number of spurts is a property of the difficulty
    distribution rather than of anything the learner does.
    """
    rng = np.random.default_rng(seed)
    first_centre = mean - 1.5 * sd
    second_centre = mean + 2.5 * sd
    # Half the words come from each group.
    n_first = n_words // 2
    n_second = n_words - n_first
    first_group = rng.normal(first_centre, sd * 0.4, size=n_first)
    second_group = rng.normal(second_centre, sd * 0.4, size=n_second)
    thresholds = np.concatenate([first_group, second_group])
    return np.maximum(thresholds, config.MIN_THRESHOLD)


def make_multifactor_thresholds(n_words: int, mean: float, sd: float,
                                seed: int, n_factors: int = 8) -> np.ndarray:
    """Build thresholds by ADDING UP many unrelated sources of difficulty.

    This is the most conceptually important generator. Each word's difficulty is
    the sum of several independent contributions -- imagine frequency, sound
    structure, how abstract the meaning is, how often the word appears alone,
    and so on. None of the individual contributions is bell-shaped: here they
    are all skewed exponential draws. But their SUM is close to bell-shaped
    anyway, because of the central limit theorem.

    The lesson: you do not have to assume a Gaussian difficulty distribution.
    You get one for free as soon as difficulty has many contributing causes.

    Args:
        n_words: how many words to generate thresholds for.
        mean: target average threshold, in time steps.
        sd: target standard deviation, in time steps.
        seed: random seed.
        n_factors: how many independent difficulty sources to add together.

    Returns:
        Array of n_words thresholds, in time steps.
    """
    rng = np.random.default_rng(seed)
    # Each factor is a skewed (exponential) contribution with mean 1 and sd 1.
    # Summing n_factors of them gives mean n_factors and sd sqrt(n_factors).
    factor_sum = np.zeros(n_words)
    for _ in range(n_factors):
        factor_sum = factor_sum + rng.exponential(scale=1.0, size=n_words)
    # Rescale that sum so it has the mean and sd the caller asked for.
    standardised = (factor_sum - n_factors) / np.sqrt(n_factors)
    thresholds = mean + sd * standardised
    return np.maximum(thresholds, config.MIN_THRESHOLD)


# A registry mapping names to generator functions. To add your own difficulty
# distribution, write a function with the same signature and add one line here.
DIFFICULTY_MAKERS = {
    "gaussian": make_gaussian_thresholds,
    "uniform": make_uniform_thresholds,
    "exponential": make_exponential_thresholds,
    "lognormal": make_lognormal_thresholds,
    "gamma": make_gamma_thresholds,
    "bimodal": make_bimodal_thresholds,
    "multifactor": make_multifactor_thresholds,
}


def make_thresholds(name: str, n_words: int, mean: float, sd: float,
                    seed: int) -> np.ndarray:
    """Look up a difficulty distribution by name and build thresholds with it.

    Args:
        name: one of the keys of DIFFICULTY_MAKERS, e.g. "gaussian".
        n_words, mean, sd, seed: passed straight to the chosen generator.

    Returns:
        Array of thresholds, in time steps.

    Raises:
        KeyError: if the name is not in the registry (the message lists the
        valid options, which is friendlier than a bare KeyError).
    """
    if name not in DIFFICULTY_MAKERS:
        valid = ", ".join(sorted(DIFFICULTY_MAKERS))
        raise KeyError("Unknown difficulty distribution %r. Valid options: %s"
                       % (name, valid))
    maker = DIFFICULTY_MAKERS[name]
    return maker(n_words, mean, sd, seed)


# ---------------------------------------------------------------------------
# Frequency-based thresholds
# ---------------------------------------------------------------------------

def thresholds_from_frequencies(frequencies: np.ndarray,
                                base: float,
                                scale: float) -> np.ndarray:
    """Turn word frequencies into time-to-acquisition thresholds.

    Frequent words should be easy, so the most frequent word gets the smallest
    threshold. The conversion is:

        threshold = base + scale * (largest log frequency - this log frequency)

    Args:
        frequencies: raw counts or rates, one per word. Must all be positive.
        base: threshold given to the single most frequent word, in time steps.
        scale: how many time steps to add per unit of log frequency lost.

    Returns:
        Array of thresholds, in time steps, same length as `frequencies`.

    Raises:
        ValueError: if any frequency is not positive (log would be undefined).
    """
    frequencies = np.asarray(frequencies, dtype=float)
    if np.any(frequencies <= 0):
        raise ValueError("All frequencies must be positive to take a logarithm.")
    log_frequencies = np.log(frequencies)
    distance_from_top = log_frequencies.max() - log_frequencies
    return base + scale * distance_from_top


def sampling_probabilities_from_frequencies(frequencies: np.ndarray) -> np.ndarray:
    """Turn frequencies into probabilities of encountering each word on a given step.

    Uses LOG frequency rather than raw frequency, which compresses the enormous
    gap between the commonest and rarest words into something a learner could
    plausibly experience.

    Args:
        frequencies: raw counts or rates, one per word. Must all be positive.

    Returns:
        Array of probabilities that sums to 1.0, same length as `frequencies`.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    if np.any(frequencies <= 0):
        raise ValueError("All frequencies must be positive to take a logarithm.")
    log_frequencies = np.log(frequencies)
    # Shift so the rarest word still has a small positive weight rather than
    # zero or a negative one.
    weights = log_frequencies - log_frequencies.min() + 1.0
    return weights / weights.sum()
