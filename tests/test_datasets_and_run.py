"""Tests for datasets.py and for the whole pipeline running end to end."""

import os
import subprocess
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)

import config
import datasets


# ---------------------------------------------------------------------------
# Synthetic frequencies
# ---------------------------------------------------------------------------

def test_zipf_frequencies_decrease():
    """Frequencies must fall as rank increases."""
    frequencies = datasets.make_zipf_frequencies(500, exponent=1.0, offset=0.0)
    assert len(frequencies) == 500
    assert np.all(np.diff(frequencies) < 0)
    assert np.all(frequencies > 0)


def test_zipf_offset_flattens_the_head():
    """A larger offset should make the top of the list less lopsided."""
    steep = datasets.make_zipf_frequencies(100, exponent=1.0, offset=0.0)
    flat = datasets.make_zipf_frequencies(100, exponent=1.0, offset=20.0)
    # Ratio of the commonest word to the tenth commonest.
    steep_ratio = steep[0] / steep[9]
    flat_ratio = flat[0] / flat[9]
    assert flat_ratio < steep_ratio


def test_zipf_rejects_bad_parameters():
    """Invalid Zipf parameters should raise rather than produce nonsense."""
    for exponent, offset in ((0.0, 0.0), (-1.0, 0.0), (1.0, -5.0)):
        try:
            datasets.make_zipf_frequencies(10, exponent, offset)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for exponent=%s offset=%s"
                                 % (exponent, offset))


def test_get_frequencies_falls_back_to_synthetic():
    """With no data files present, we must still get usable frequencies.

    Crucially the source label has to say so, so a synthetic run can never be
    mistaken for a real one.
    """
    with tempfile.TemporaryDirectory() as empty_directory:
        for register in ("child", "adult"):
            data = datasets.get_frequencies(register, empty_directory, 200)
            assert len(data["frequencies"]) == 200
            assert data["is_real"] is False
            assert "SYNTHETIC" in data["source"]


def test_get_frequencies_rejects_unknown_register():
    """Only 'child' and 'adult' are valid registers."""
    try:
        datasets.get_frequencies("martian", "data", 10)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a ValueError for an unknown register.")


def test_child_and_adult_synthetic_lists_differ():
    """The two synthetic registers must not be identical."""
    with tempfile.TemporaryDirectory() as empty_directory:
        child = datasets.get_frequencies("child", empty_directory, 500)
        adult = datasets.get_frequencies("adult", empty_directory, 500)
        assert not np.array_equal(child["frequencies"], adult["frequencies"])


# ---------------------------------------------------------------------------
# Real-data loaders, exercised with small synthetic files
# ---------------------------------------------------------------------------

def test_mrc_loader_reads_a_well_formed_file():
    """The MRC loader should pick out words and frequencies and sort them."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "mrc.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("word,kf_freq,nphon\n")
            handle.write("the,69971,2\n")
            handle.write("dog,75,3\n")
            handle.write("cat,23,3\n")
        words, frequencies = datasets.load_mrc_frequencies(path, n_words=10)
        assert words == ["the", "dog", "cat"]
        assert frequencies[0] == 69971.0
        assert np.all(np.diff(frequencies) < 0)


def test_mrc_loader_handles_duplicate_words():
    """MRC lists a word once per sense; we keep one entry per word."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "mrc.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("word,kf_freq\n")
            handle.write("run,100\n")
            handle.write("run,100\n")
            handle.write("jump,50\n")
        words, frequencies = datasets.load_mrc_frequencies(path, n_words=10)
        assert words == ["run", "jump"]
        assert len(frequencies) == 2


def test_mrc_loader_skips_unusable_rows():
    """Rows with missing or non-numeric frequencies are ignored, not fatal."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "mrc.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("word,kf_freq\n")
            handle.write("good,100\n")
            handle.write("bad,\n")
            handle.write("worse,notanumber\n")
            handle.write("zero,0\n")
        words, frequencies = datasets.load_mrc_frequencies(path, n_words=10)
        assert words == ["good"]


def test_mrc_loader_reports_missing_file():
    """A missing file should be a clear FileNotFoundError."""
    try:
        datasets.load_mrc_frequencies("/no/such/file.csv", 10)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected a FileNotFoundError.")


def test_mrc_loader_reports_wrong_columns():
    """A file without recognisable columns should say so clearly."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "mrc.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("alpha,beta\n1,2\n")
        try:
            datasets.load_mrc_frequencies(path, 10)
        except ValueError as error:
            assert "columns" in str(error).lower()
        else:
            raise AssertionError("Expected a ValueError about columns.")


def test_wordbank_loader_averages_within_age():
    """Several children of the same age should be averaged."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "wordbank.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("age,production\n")
            handle.write("16,10\n")
            handle.write("16,20\n")
            handle.write("18,60\n")
        norms = datasets.load_wordbank_norms(path)
        assert list(norms["ages"]) == [16, 18]
        assert norms["mean_words_produced"][0] == 15.0
        assert norms["mean_words_produced"][1] == 60.0
        assert list(norms["n_children"]) == [2, 1]


def test_mrc_loader_picks_kf_column_from_the_real_header():
    """Guard against latching onto the wrong frequency column.

    The real MRC export has THREE frequency columns side by side: Kucera-Francis
    written frequency, Thorndike-Lorge, and Brown verbal. We want the first.
    This test uses the genuine header line so that a future change to the
    column-matching logic cannot silently start reading the wrong one.
    """
    real_header = (
        "Word,Number of Letters,Number of Phonemes,Number of Syllables,"
        "KF Written Frequency,KF Number of Categories,KF Number of Samples,"
        "Thorndike-Lorge Frequency,Brown Verbal Frequency,Familiarity,"
        "Concreteness,Imageability,Meaningfulness: Coloradao Norms,"
        "Meaningfulness: Pavio Norms,Age of Acquisition Rating,Word Type,"
        "Comprehensive Syntactic Category,Common Part of Speech,"
        "Morphemic status,Contextual Status,Pronunciation Variability,"
        "Capitalization,Irregular Plural,Stress-Marked Phonetic Transcription,"
        "Syllabified Phonetic Transcription,Stress Pattern"
    )
    # KF frequency is deliberately NOT in descending order relative to the other
    # frequency columns, so reading the wrong one gives a different ranking.
    rows = [
        "&ARRY,5,0,0,0,0,0,0,0,0,0,0,0,0,0, ,N, ,A,S, , , ,,,",
        "THE,3,1,1,69971,15,500,10,5,0,0,0,0,0,0, ,N, ,A,S, , , ,,,",
        "DOG,3,3,1,75,12,50,99999,88888,0,596,610,0,0,0, ,N, ,A,S, , , ,,,",
        "DOG,3,3,1,75,12,50,99999,88888,0,596,610,0,0,0, ,N, ,A,S, , , ,,,",
    ]

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "mrc.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(real_header + "\n")
            handle.write("\n".join(rows) + "\n")

        words, frequencies = datasets.load_mrc_frequencies(path, n_words=100)

        # "the" must outrank "dog" on KF frequency. If the loader had picked
        # Thorndike-Lorge or Brown instead, "dog" would come first.
        assert words == ["the", "dog"], words
        assert frequencies[0] == 69971.0
        assert frequencies[1] == 75.0


def test_mrc_loader_drops_non_words():
    """Entries like '&arry' are dropped; ordinary hyphenated words are kept."""
    assert datasets._looks_like_a_word("the")
    assert datasets._looks_like_a_word("well-known")
    assert datasets._looks_like_a_word("don't")
    assert not datasets._looks_like_a_word("&arry")
    assert not datasets._looks_like_a_word("a1")
    assert not datasets._looks_like_a_word("-x")
    assert not datasets._looks_like_a_word("")


def test_get_wordbank_norms_returns_none_when_absent():
    """A missing Wordbank file is not an error, it just skips that figure."""
    with tempfile.TemporaryDirectory() as empty_directory:
        assert datasets.get_wordbank_norms(empty_directory) is None


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_full_pipeline_runs_and_writes_everything():
    """`python run_all.py --quick` must succeed and produce all its outputs.

    This is the single test that most reliably tells you the project as a whole
    is working.
    """
    with tempfile.TemporaryDirectory() as output_directory:
        completed = subprocess.run(
            [sys.executable, "run_all.py", "--quick",
             "--outdir", output_directory],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)

        assert completed.returncode == 0, (
            "run_all.py failed:\nSTDOUT:\n%s\nSTDERR:\n%s"
            % (completed.stdout, completed.stderr))

        expected_files = [
            "fig1_difficulty_distribution.png",
            "fig2_baseline_growth_and_rate.png",
            "fig3_benefit_and_cost.png",
            "fig4_frequency_models.png",
            "fig5_sampling_model.png",
            "fig6_distribution_comparison.png",
            "results.csv",
            "run_log.txt",
        ]
        for filename in expected_files:
            full_path = os.path.join(output_directory, filename)
            assert os.path.exists(full_path), "missing output: %s" % filename
            assert os.path.getsize(full_path) > 0, "empty output: %s" % filename
