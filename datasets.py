"""
Where the word data comes from.

The model needs two kinds of input, and both have a real version and a
synthetic version:

  Word frequencies   -> used to build difficulty from how often words occur.
                        Real:      the MRC Psycholinguistic Database.
                        Synthetic: a Zipf-Mandelbrot law (see below).

  Child vocabulary   -> real growth curves to compare the model against.
     norms               Real:      Wordbank CDI norms.
                        Synthetic: none. If you do not have the file, the
                                   comparison figure is simply skipped.

Nothing in this project REQUIRES the real data. The synthetic frequencies have
the properties the model cares about, which is the point of the exercise: you
are studying what follows from the shape of a distribution, not from the
particular words of English.

Every loader returns data plus a short label saying where it came from, and that
label is printed and drawn on the figures, so a synthetic run can never be
mistaken for a real one.
"""

import os

import numpy as np

import config


# ---------------------------------------------------------------------------
# Synthetic frequencies
# ---------------------------------------------------------------------------

def make_zipf_frequencies(n_words: int, exponent: float,
                          offset: float) -> np.ndarray:
    """Build a synthetic frequency list following a Zipf-Mandelbrot law.

    Real vocabularies have a few very common words and a long tail of rare ones.
    The Zipf-Mandelbrot law captures that with two knobs:

        frequency(rank) = 1 / (rank + offset) ** exponent

    `exponent` controls how steeply frequency falls off: bigger means a more
    lopsided vocabulary. `offset` flattens the top of the list: bigger means the
    first few words are less dominant relative to each other.

    Args:
        n_words: how many words to generate.
        exponent: fall-off steepness. Must be positive.
        offset: flattening of the head of the distribution. Must be >= 0.

    Returns:
        Array of n_words frequencies, in arbitrary units, largest first.
    """
    if exponent <= 0:
        raise ValueError("exponent must be positive.")
    if offset < 0:
        raise ValueError("offset must not be negative.")
    ranks = np.arange(1, n_words + 1, dtype=float)
    return 1.0 / (ranks + offset) ** exponent


# ---------------------------------------------------------------------------
# Real frequencies: MRC Psycholinguistic Database
# ---------------------------------------------------------------------------

def load_mrc_frequencies(path: str, n_words: int) -> tuple:
    """Read word frequencies out of an MRC Psycholinguistic Database file.

    The MRC database gives, for around 150,000 entries, a written-frequency
    count alongside properties like number of phonemes, concreteness and
    familiarity. We use the frequency column here; see load_mrc_properties for
    the rest.

    The loader is deliberately tolerant about column naming, because the MRC
    data is distributed in several formats. It looks for a column whose name
    contains "word" and one whose name contains "freq".

    Args:
        path: path to a CSV file of MRC data.
        n_words: how many of the most frequent words to keep.

    Returns:
        Tuple of (words, frequencies), where words is a list of strings and
        frequencies is a numpy array, sorted most frequent first.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if no usable word and frequency columns can be found.
    """
    import csv

    if not os.path.exists(path):
        raise FileNotFoundError("No MRC file at %s" % path)

    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("MRC file %s has no header row." % path)
        word_column = _find_column(reader.fieldnames, ["word"])
        # "kf_written_frequency" is the exact column name in the standard MRC
        # export and is listed first so we never accidentally pick up one of the
        # other frequency columns (Thorndike-Lorge, Brown verbal) that sit
        # alongside it in the same file.
        freq_column = _find_column(reader.fieldnames,
                                   ["kf_written_frequency", "kf_freq", "kffreq",
                                    "written_frequency", "frequency", "freq"])
        if word_column is None or freq_column is None:
            raise ValueError(
                "Could not find word and frequency columns in %s. "
                "Columns present: %s" % (path, reader.fieldnames))

        pairs = []
        for row in reader:
            word = (row.get(word_column) or "").strip().lower()
            raw_frequency = (row.get(freq_column) or "").strip()
            if not word or not raw_frequency:
                continue
            # MRC contains entries like "&ARRY" and abbreviations. Keep only
            # ordinary words: letters, optionally with one internal hyphen or
            # apostrophe.
            if not _looks_like_a_word(word):
                continue
            try:
                frequency = float(raw_frequency)
            except ValueError:
                continue
            if frequency > 0:
                pairs.append((word, frequency))

    if not pairs:
        raise ValueError("No usable rows found in %s." % path)

    # Keep one entry per word (the MRC file lists a word once per sense).
    best_for_word = {}
    for word, frequency in pairs:
        if word not in best_for_word or frequency > best_for_word[word]:
            best_for_word[word] = frequency

    ranked = sorted(best_for_word.items(), key=lambda pair: pair[1], reverse=True)
    ranked = ranked[:n_words]
    words = [word for word, _ in ranked]
    frequencies = np.array([frequency for _, frequency in ranked], dtype=float)
    return words, frequencies


def _looks_like_a_word(text: str) -> bool:
    """Decide whether a database entry is an ordinary word we want to keep.

    The MRC database includes entries that are not words in the ordinary sense:
    fragments like "&arry", abbreviations, and entries with embedded markup. We
    keep only strings made of letters, allowing one internal hyphen or
    apostrophe (so "well-known" and "don't" survive).

    Args:
        text: the candidate word, already lowercased and stripped.

    Returns:
        True if the entry looks like an ordinary word.
    """
    if not text:
        return False
    if not text[0].isalpha() or not text[-1].isalpha():
        return False
    for character in text:
        if not (character.isalpha() or character in "-'"):
            return False
    return True


def _find_column(fieldnames: list, wanted_substrings: list):
    """Find the first column whose name contains one of the wanted substrings.

    Args:
        fieldnames: the column names read from the file.
        wanted_substrings: candidate substrings, tried in order of preference.

    Returns:
        The matching column name, or None if nothing matched.
    """
    lowered = {name.lower().replace(" ", "_"): name for name in fieldnames}
    for wanted in wanted_substrings:
        for lowered_name, original_name in lowered.items():
            if wanted in lowered_name:
                return original_name
    return None


# ---------------------------------------------------------------------------
# The loader the rest of the project actually calls
# ---------------------------------------------------------------------------

def get_frequencies(register: str, data_dir: str, n_words: int) -> dict:
    """Get a frequency list for one speech register, real if available.

    Tries the real MRC file first; falls back to synthetic frequencies with a
    clearly marked label.

    Args:
        register: either "child" or "adult". These select different synthetic
            parameters, and only "adult" has a real data source (MRC is adult
            written English).
        data_dir: folder to look for data files in.
        n_words: how many words to return.

    Returns:
        Dictionary with:
          "words"       : list of word strings, or None for synthetic data.
          "frequencies" : numpy array of frequencies, most frequent first.
          "source"      : short human-readable label of where this came from.
          "is_real"     : True if loaded from a real dataset.
    """
    if register not in ("child", "adult"):
        raise ValueError("register must be 'child' or 'adult', got %r" % register)

    if register == "adult":
        mrc_path = os.path.join(data_dir, "mrc.csv")
        try:
            words, frequencies = load_mrc_frequencies(mrc_path, n_words)
            return {"words": words, "frequencies": frequencies,
                    "source": "MRC Psycholinguistic Database",
                    "is_real": True}
        except (FileNotFoundError, ValueError):
            pass  # fall through to synthetic

    if register == "child":
        exponent = config.CHILD_ZIPF_EXPONENT
        offset = config.CHILD_ZIPF_OFFSET
    else:
        exponent = config.ADULT_ZIPF_EXPONENT
        offset = config.ADULT_ZIPF_OFFSET

    frequencies = make_zipf_frequencies(n_words, exponent, offset)
    return {"words": None, "frequencies": frequencies,
            "source": "SYNTHETIC (Zipf-Mandelbrot, exponent=%.2f, offset=%.0f)"
                      % (exponent, offset),
            "is_real": False}


# ---------------------------------------------------------------------------
# Real child vocabulary norms: Wordbank
# ---------------------------------------------------------------------------

def load_wordbank_norms(path: str) -> dict:
    """Read child vocabulary norms out of a Wordbank export.

    Wordbank publishes, for each child, an age in months and a count of words
    the child produces. We average those counts within each age.

    Like the MRC loader, this is tolerant about column names: it looks for a
    column containing "age" and one containing "produc" or "vocab".

    Args:
        path: path to a CSV file exported from Wordbank.

    Returns:
        Dictionary with "ages" (months) and "mean_words_produced", both numpy
        arrays sorted by age, plus "n_children" per age.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if suitable columns cannot be found.
    """
    import csv

    if not os.path.exists(path):
        raise FileNotFoundError("No Wordbank file at %s" % path)

    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Wordbank file %s has no header row." % path)
        age_column = _find_column(reader.fieldnames, ["age"])
        produced_column = _find_column(reader.fieldnames,
                                       ["production", "produc", "vocab", "words"])
        if age_column is None or produced_column is None:
            raise ValueError(
                "Could not find age and production columns in %s. "
                "Columns present: %s" % (path, reader.fieldnames))

        totals = {}
        counts = {}
        for row in reader:
            try:
                age = int(float(row[age_column]))
                produced = float(row[produced_column])
            except (TypeError, ValueError):
                continue
            totals[age] = totals.get(age, 0.0) + produced
            counts[age] = counts.get(age, 0) + 1

    if not totals:
        raise ValueError("No usable rows found in %s." % path)

    ages = sorted(totals)
    means = [totals[age] / counts[age] for age in ages]
    n_children = [counts[age] for age in ages]
    return {"ages": np.array(ages),
            "mean_words_produced": np.array(means),
            "n_children": np.array(n_children)}


def get_wordbank_norms(data_dir: str):
    """Load Wordbank norms if the file is present, otherwise return None.

    Args:
        data_dir: folder to look in for wordbank.csv.

    Returns:
        The dictionary from load_wordbank_norms, or None if unavailable.
    """
    path = os.path.join(data_dir, "wordbank.csv")
    try:
        return load_wordbank_norms(path)
    except (FileNotFoundError, ValueError):
        return None
