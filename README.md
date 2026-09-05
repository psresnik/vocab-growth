# Where does the vocabulary spurt come from?

## Contents

- [Install](#install)
- [Run it](#run-it)
- [Test it](#test-it)
- [How the model works](#how-the-model-works)
- [Understanding the code](#understanding-the-code)
- [What each result is, and where it comes from](#what-each-result-is-and-where-it-comes-from)
- [The files](#the-files)
- [Extending it](#extending-it)
- [Questions worth chasing](#questions-worth-chasing)
- [Honest limitations](#honest-limitations)

Somewhere around eighteen months, most children start learning words much faster
than they were before. This is usually called the vocabulary spurt or the
vocabulary explosion, and the obvious explanation is that something changes in
the child: a new insight, a new strategy, a new mechanism switching on.

This project asks whether you need any of that.

The model here has no insight, no strategy and no mechanism that switches on. It
accumulates evidence for every word at a completely constant rate from beginning
to end. The only interesting thing about it is that some words take longer to
learn than others. That alone turns out to be enough to produce a curve that
looks exactly like a vocabulary spurt.

The approach is due to Bob McMurray (2007), *Defusing the childhood vocabulary
explosion*, Science 317, 631. This is not a replication of that paper: the
parameters and the word data are our own, and the point is to get familiar with
the style of argument rather than to reproduce specific published numbers.

---

## Install

With pip:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

With conda:

```bash
conda env create -f environment.yml
conda activate vocab-growth
```

You need Python 3.9 or newer, numpy and matplotlib. That is all. There is
nothing to download and no network access required.

---

## Run it

```bash
python run_all.py
```

Takes about five seconds and writes six figures, a results table and a run log
into `output/`.

```bash
python run_all.py --quick              # smaller and faster, for checking things work
python run_all.py --seed 7             # different random numbers
python run_all.py --n-words 50000      # bigger vocabulary
python run_all.py --outdir myrun       # write somewhere else
python run_all.py --help               # all options
```

## Test it

```bash
python run_tests.py                    # no extra packages needed
python run_tests.py -v                 # show every test name
python run_tests.py --only sampling    # just the tests matching "sampling"
```

or, if you have pytest installed, plain `pytest` does the same thing.

There are 60 tests and they take well under a minute. If they all pass, the
model is behaving the way it is supposed to.

---

## How the model works

Every word has a **threshold**: how much evidence the learner needs before that
word counts as known. Different words have different thresholds, because words
differ in how often they come up, how easy they are to say, how concrete their
meaning is, and so on.

On every time step, every word the learner does not yet know gains exactly one
point of evidence. No exceptions, no speeding up, no strategy. When a word's
points reach its threshold, it is learned.

That is the whole model. The number of words known after *t* steps is just the
number of thresholds less than or equal to *t*.

Now think about what that means. If word difficulty is bell-shaped -- a few easy
words, lots of moderate ones, a few very hard ones -- then in the early period
you are collecting from the thin left tail of the bell, and you learn very few
words. Later you reach the fat middle, and words start arriving in a flood. The
learner has not changed at all. The *distribution* changed what the learner ran
into.

The spurt, in other words, is a fact about the vocabulary, not about the child.

Two follow-ups sharpen the point:

- If learning each word makes the remaining words **easier**, growth
  accelerates. If learning each word makes the remaining words **harder**,
  growth still accelerates. So observing a spurt tells you nothing about which
  is happening.
- Even if you make every word **equally difficult**, and vary only how often
  each word is encountered, you still get a spurt.

---
## Understanding the code

The code here was created using Claude Opus 5 in September, 2026.
Claude was instructed to write code in a way that would be reasonably
readable for people relatively new to python, but of course reading
and understanding someone else's code is something that takes
practice. One thing that can be useful is asking Claude itself to
explain the code.  Here's an example with a prompt that seems to work
well:

```
Explain each aspect of the following function to someone who is just beginning to learn python.

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
```

Here's the explanation it generates.

### The function signature

```python
def make_gaussian_thresholds(n_words: int, mean: float, sd: float,
                             seed: int) -> np.ndarray:
```

- `def` starts a function definition — this is how you teach Python a new command.
- `make_gaussian_thresholds` is the name. Good Python style uses lowercase words separated by underscores, and a name that says what the function *does*.
- Inside the parentheses are the **parameters** — the inputs the function needs. There are four: `n_words`, `mean`, `sd`, `seed`.
- The `: int`, `: float` after each parameter are **type hints**. They're documentation, not enforcement — Python won't stop you from passing a string instead of an int. They just tell a reader (and some editors) what kind of value is expected. `int` means whole number, `float` means a number that can have a decimal point.
- `-> np.ndarray` after the parentheses is a type hint for the **return value**: this function hands back a numpy array (`np` is the standard nickname for the numpy library, `ndarray` is numpy's array type).

### The docstring

```python
    """Draw thresholds from a normal (bell-shaped) distribution.
    ...
    """
```

Everything between the triple quotes is a **docstring** — a string literal placed right after `def` whose only job is documentation. It's not a comment (which uses `#`); it's an actual string that Python stores, so tools can show it when someone types `help(make_gaussian_thresholds)`. The convention used here — a one-line summary, then `Args:` describing each parameter, then `Returns:` describing the output — is a common style (Google-style docstrings), not a Python requirement.

### The body

```python
    rng = np.random.default_rng(seed)
```

This creates a **random number generator** object and stores it in the variable `rng`. Handing it a `seed` (just a number) means: "start from this exact point in the sequence of random numbers." Run this with the same seed twice and you get identical output — that's what "reproducible" means in this context. Without a seed, you'd get different random numbers every time you ran the program.

```python
    thresholds = rng.normal(loc=mean, scale=sd, size=n_words)
```

`.normal(...)` is a **method** — a function that belongs to the `rng` object — that draws random numbers from a normal (bell-curve) distribution.

- `loc=mean` and `scale=sd` are **keyword arguments**: instead of relying on argument order, you name which parameter you're setting. `loc` is numpy's name for the center of the bell curve (the mean); `scale` is its name for the spread (the standard deviation).
- `size=n_words` says how many numbers to generate at once.
- The result isn't one number — it's a whole numpy **array** of `n_words` numbers, stored in the variable `thresholds`.

### The comment

```python
    # np.maximum compares every element against the floor value at once; the
    # loop-over-words version would be: [max(t, MIN) for t in thresholds]
```

Lines starting with `#` are comments — plain English notes for humans, ignored entirely by Python. This one explains *why* the next line is written the way it is, and translates it into a form a beginner might find more familiar (a list comprehension using Python's built-in `max`).

### The return line

```python
    return np.maximum(thresholds, config.MIN_THRESHOLD)
```

- `config.MIN_THRESHOLD` reaches into a different file, `config.py`, and grabs the constant named `MIN_THRESHOLD` defined there. The dot (`.`) means "look inside this module for this name."
- `np.maximum(a, b)` compares two things **element-by-element** and keeps whichever is larger at each position. Here, `thresholds` is a whole array and `MIN_THRESHOLD` is a single number; numpy automatically compares every entry in the array against that one number (this stretching-to-fit is called *broadcasting*). The comment above already explained the plain-loop equivalent.
- `return` sends this final array back to whoever called the function — it's the function's output.

**Big picture:** the function makes a batch of random "difficulty" numbers clustered around `mean` with spread `sd`, then makes sure none of them dip below a minimum allowed value, and hands the result back as one array.

---

## What each result is, and where it comes from

| Result | Produced by | Output file |
|---|---|---|
| Difficulty distribution, with two equal-width windows shaded to show the later one contains more words | `run_baseline` | `fig1_difficulty_distribution.png` |
| Growth curve and learning rate for the baseline model: speeds up, then slows down | `run_baseline` | `fig2_baseline_growth_and_rate.png` |
| Growth when learning a word helps, hinders, or does neither | `run_benefit_and_cost` | `fig3_benefit_and_cost.png` |
| Growth when difficulty is derived from word frequency, for two registers | `run_frequency_models` | `fig4_frequency_models.png` |
| Growth when every word is equally hard and only encounter rate varies | `run_sampling_model` | `fig5_sampling_model.png` |
| Seven difficulty distributions side by side, showing which produce acceleration | `run_distribution_comparison` | `fig6_distribution_comparison.png` |
| Model shape against real child vocabulary norms (only if you supply Wordbank data) | `figures.plot_model_against_norms` | `fig7_model_vs_norms.png` |
| Summary numbers for every run above | `write_results_table` | `results.csv` |
| Seed, settings, library versions, data sources used | `write_run_log` | `run_log.txt` |

---

## The files

```
config.py       every tunable number, with units and an explanation
difficulty.py   building word thresholds: seven distributions, plus frequency-based
simulation.py   the two learning loops
analysis.py     measuring the shape of a growth curve
datasets.py     loading real word data, or generating synthetic data instead
figures.py      drawing
run_all.py      runs everything and writes the outputs
run_tests.py    runs the tests without needing pytest
tests/          60 tests
experiments/    two worked examples of extending the project
data/           optional real datasets; see data/README.md
```

Every random function takes an explicit `seed`, so the same command always gives
the same numbers. There are no classes anywhere: everything is plain functions
operating on numpy arrays and dictionaries.

---

## Extending it

**Change a number.** Everything tunable is in `config.py` with a comment saying
what it means and what units it is in.

**Add a difficulty distribution.** Write a function with the same signature as
the ones in `difficulty.py` and add one line to the `DIFFICULTY_MAKERS`
dictionary. It will then work everywhere, including in the comparison figure.
`experiments/example_new_distribution.py` does exactly this.

**Sweep a parameter.** `experiments/example_parameter_sweep.py` shows the
pattern: loop over values, collect a summary number for each, plot the summaries.

Both examples run as they are:

```bash
python experiments/example_new_distribution.py
python experiments/example_parameter_sweep.py
```

---

## Questions worth chasing

- The model predicts that growth **decelerates** at the end, once the easy words
  are used up. Does that actually happen in children? What would you need to
  measure to find out?
- The `exponential` distribution does not produce acceleration. Look at its
  shape and work out why, then predict which other distributions will fail
  before you run them.
- `bimodal` difficulty gives two spurts. Is there any evidence for more than one
  spurt in real vocabulary growth?
- The model gives every word the same constant learning rate. What would change
  if the rate itself varied between words? Would the argument survive?
- The frequency-based version uses log frequency. What happens with raw
  frequency instead, and which is the more defensible choice?

---

## Honest limitations

Read `Issues.md`. It records every judgement call made while building this,
what the alternatives were, and what the consequences are. The most important
ones:

- The default synthetic frequency lists were chosen to illustrate a pattern, not
  measured from any corpus. The difference between the two registers in
  `fig4` for the synthetic output is a property of parameters we picked.
- "Time steps" are not calibrated to real time, so nothing here predicts *when*
  a spurt should happen, only what shape it should have.
- Showing that a simple model *can* produce a spurt does not show that this is
  how children actually work. It shows that the spurt is not by itself evidence
  for anything more complicated.
