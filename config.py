"""
All tunable numbers for the vocabulary-growth model live here.

Keeping every constant in one file means you can run new experiments by editing
this file (or by passing arguments to the simulation functions) without touching
the simulation code itself.

Units used throughout the project:
  - "time step"  : one tick of the simulation clock. Not calibrated to real time.
  - "point"      : one unit of evidence accumulated toward learning a word.
  - "threshold"  : how many points a word needs before it counts as learned.
                   A word's threshold IS its time-to-acquisition when every word
                   gains exactly one point per step.
"""

# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

# How many words the simulated learner is trying to acquire.
N_WORDS = 10000

# How many time steps to run the baseline simulations for.
N_STEPS = 7000

# Shape of the time-to-acquisition distribution for the baseline model.
# A Gaussian centred here means most words take about this long to learn.
THRESHOLD_MEAN = 4000.0
THRESHOLD_SD = 680.0

# Random seed. Every function that samples takes a seed argument; this is the
# default so that a plain `python run_all.py` is reproducible.
SEED = 20240617

# ---------------------------------------------------------------------------
# Benefit / cost coupling between words
# ---------------------------------------------------------------------------
# When a word is learned, the thresholds of all still-unlearned words shift by
# this many points per newly learned word, on that step.
#   negative -> "benefit": learning a word makes the remaining words easier
#   positive -> "cost":    learning a word makes the remaining words harder
BENEFIT_SHIFT_PER_WORD = -0.1
COST_SHIFT_PER_WORD = +0.1

# Thresholds are never allowed below this value, so a word can never be learned
# before it has accumulated at least a little evidence.
MIN_THRESHOLD = 1.0

# ---------------------------------------------------------------------------
# Frequency-based difficulty
# ---------------------------------------------------------------------------
# How many words to take from a frequency list.
N_FREQUENT_WORDS = 2000

# Thresholds are built from log frequency as:
#     threshold = FREQ_BASE + FREQ_SCALE * (max_log_freq - log_freq)
# so the most frequent word has threshold FREQ_BASE and rarer words take longer.
FREQ_BASE = 3000.0
FREQ_SCALE = 800.0

# Synthetic frequency lists follow a Zipf-Mandelbrot law:
#     frequency(rank) = 1 / (rank + offset) ** exponent
# The two registers use different parameters so that the "child-directed" list
# has a flatter head (a few words that dominate) and a steeper tail.
# See Issues.md #4: these values are chosen to illustrate a pattern, they are
# not measured from any corpus.
CHILD_ZIPF_EXPONENT = 1.8
CHILD_ZIPF_OFFSET = 25.0
ADULT_ZIPF_EXPONENT = 1.0
ADULT_ZIPF_OFFSET = 0.0

# ---------------------------------------------------------------------------
# Fixed-threshold sampling model
# ---------------------------------------------------------------------------
# In this model every word is equally hard, but words are encountered at
# different rates. A word is learned once it has been encountered this often.
SAMPLING_THRESHOLD = 10

# Safety cap so the sampling simulation always terminates.
MAX_SAMPLING_STEPS = 400000

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
# Width, in time steps, of the window used to measure "words learned per unit
# time" when checking for acceleration.
RATE_WINDOW = 100

# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
DATA_DIR = "data"
OUTPUT_DIR = "output"
