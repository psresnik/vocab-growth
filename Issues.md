# Issues Log

Judgement calls made while building this project, what the alternatives were,
and what difference each one makes to the results.

Format for each entry: what forced a decision, what the options were, what was
chosen, why, and what a reader should watch out for as a result.

---

## Issue 1: Replication was abandoned in favour of a teaching implementation

**Status:** Resolved

**Context:** This started as an attempt to replicate McMurray (2007) exactly.
That required child-directed speech frequencies from CHILDES. The `childesr` R
package connects to a MySQL server by raw IP on port 3306, which was unreachable
from the development machine; the migration path (childes-db on Redivis) needs an
account and an API token, which cannot be obtained unattended. Several hours went
into diagnosing this before the goal changed.

**Options considered:**
- Keep chasing the data. Pro: exact replication. Con: blocked on infrastructure
  outside our control, with no guarantee of success.
- Replicate with substitute corpora. Pro: real data. Con: the numbers would not
  match the paper anyway, so the "replication" label would be misleading.
- **Build a teaching implementation that produces results *like* the paper, with
  synthetic data by default and real data optional.** Pro: no external
  dependency, runs anywhere, and for a teaching goal the mechanism matters more
  than matching published figures. Con: cannot be cited as a replication.

**Decision:** The third. Nothing in the project claims to reproduce McMurray's
specific numbers, and the README says so explicitly.

**Impact:** Do not use this code to check McMurray's arithmetic. Use it to
understand the argument.

---

## Issue 2: Where the acceleration test's definition came from

**Status:** Resolved, after the first version failed

**Context:** "Does this run accelerate?" needs an operational definition. The
first implementation compared words learned in the second quarter of the run
against the first quarter. It gave the wrong answer twice: the sampling model
was scored as non-accelerating when its learning rate visibly rises to a peak
and then falls, and the multifactor distribution scored a marginal 1.09. Both
failures had the same cause: a long right tail stretches the run, so the "first
quarter" already contains the entire spurt.

**Options considered:**
- Keep quarters, loosen the thresholds until the tests pass. Pro: least work.
  Con: fits the test to the answer, which is the failure mode this whole project
  is about.
- Compare quarters but over a trimmed period (1% to 99% of final vocabulary).
  Pro: simple fix, handles the multifactor case (ratio rose to 1.70). Con: still
  scored the sampling model as non-accelerating (0.81), because its spurt is
  short relative to its long tail.
- **Compare the peak learning rate against the rate at the start of learning,
  over the trimmed period, requiring the peak to be at least 1.5x the start and
  to occur after the opening tenth.** Pro: directly expresses "the learner got
  faster"; separates the cases cleanly. Con: two parameters (1.5 and the tenth)
  rather than none.

**Decision:** The third. Measured values: gaussian 14.7, gamma 7.6, lognormal
7.0, multifactor 3.8, bimodal 3.4, sampling 2.2 (all accelerating); exponential
1.3 and uniform 1.2 (both not). The gap between 2.2 and 1.3 is wide enough that
the exact threshold is not doing much work.

**Impact:** `acceleration_ratio` in `results.csv` means peak rate divided by
starting rate. It is not comparable to a quartile ratio. The quartile counts are
still reported in the table for anyone who wants them, but nothing depends on
them.

---

## Issue 3: The exponential distribution is the negative control

**Status:** Resolved

**Context:** A model that produces acceleration under every condition would
demonstrate nothing. There has to be a case where acceleration fails, and it has
to fail for a principled reason.

**Decision:** The exponential distribution has many easy words and few hard ones,
the exact inverse of the pattern that generates acceleration. It is included in
the registry, it is asserted *not* to accelerate in the test suite, and it is
labelled as such in the comparison figure. Uniform difficulty is a second,
weaker control: flat difficulty gives straight-line growth.

**Impact:** If a change to the model ever makes the exponential case accelerate,
something is broken. That test is the most informative one in the suite.

---

## Issue 4: The two synthetic frequency registers are illustrative, not empirical

**Status:** Resolved with caveat — read this one before drawing conclusions
from figure 4

**Context:** Real child-directed and adult-directed speech have different
frequency profiles, and McMurray reports that child-directed speech gives faster
early learning while adult-directed speech catches up later. To show that
pattern without a corpus, the shapes have to be constructed.

Two Zipf distributions with different exponents cannot produce a crossover: under
the log-frequency-to-threshold mapping, a steeper exponent is uniformly slower.
Producing the crossover requires the child-directed list to have a *flatter head*
(so early words are learned sooner) and a *steeper tail* (so late words are
learned later). That needs the Zipf-Mandelbrot form with two parameters.

**Options considered:**
- Drop the two-register comparison entirely. Pro: nothing to misinterpret. Con:
  loses a result that shows distribution shape, not just spread, matters.
- Fit the parameters to a real corpus. Pro: honest. Con: reintroduces the data
  dependency this project exists to avoid.
- **Choose parameters that produce the qualitative pattern, label them clearly
  as chosen, and stamp `SYNTHETIC` on the source string.** Pro: keeps the
  teaching point, cannot be mistaken for data. Con: a reader who skims might
  still over-read the figure.

**Decision:** The third, with child (exponent 1.8, offset 25) and adult
(exponent 1.0, offset 0). Measured result: the child list leads throughout the
middle (79 vs 12 words known at step 5000; 812 vs 518 at step 8000) but finishes
second (9272 vs 9081 steps to complete).

**Impact:** **The crossover in figure 4 is a consequence of parameters we chose.
It is not evidence about English.** It shows that a crossover *can* arise from
distribution shape alone, which is a claim about mechanism, not about children.
Supplying a real MRC file replaces the adult list with real data; the child list
stays synthetic either way.

---

## Issue 5: MRC is written English, not speech to children

**Status:** Unresolved by design — documented

**Context:** The MRC Psycholinguistic Database carries Kučera-Francis frequency
counts, which come from written American English. The model wants the frequency
of words *as a child encounters them*.

**Decision:** MRC is offered as the optional real frequency source for the
"adult" register only, and `data/README.md` states the limitation plainly. No
real child-directed source is offered, because obtaining one is precisely the
problem that ended the replication attempt.

**Impact:** A run with real MRC data is "difficulty derived from a real
word-frequency distribution", not "difficulty derived from what children hear".
The distinction matters for any conclusion drawn about actual language learning.

---

## Issue 6: Time steps are not calibrated to real time

**Status:** Unresolved by design — documented

**Context:** The model's clock has no units. Mapping steps to months would need
an argument about how much linguistic experience a child gets per unit time,
which the model does not contain.

**Decision:** Leave it uncalibrated and say so. The Wordbank comparison figure
rescales both curves onto a 0-to-1 axis and compares shape only.

**Impact:** Nothing here predicts *when* a spurt should happen, or how long it
should last, only what shape it should have. Any claim about age is out of scope.

---

## Issue 7: Threshold clipping distorts results at large spreads

**Status:** Resolved — surfaced deliberately as a teaching point

**Context:** Thresholds are clipped at `MIN_THRESHOLD` so that no word can be
learned before accumulating evidence. When the spread of the difficulty
distribution approaches its mean, a substantial part of the bell curve falls
below zero and piles up at the floor. Those words are all learned on step 1,
which inflates the measured starting rate and suppresses the acceleration ratio.
This is visible in the parameter sweep: the ratio holds between 10 and 15 for
spreads from 200 to 1500, then drops to 4.8 at spread 2000.

**Options considered:**
- Silently rescale the distribution to avoid the floor. Pro: smooth-looking
  results. Con: hides a real modelling constraint and teaches students that
  results are never artefacts.
- Raise an error for parameter combinations that clip heavily. Pro: safe. Con:
  blocks legitimate exploration.
- **Leave the behaviour in place and call it out in the sweep's own output.**
  Pro: students meet a genuine artefact in a controlled setting and are told how
  to diagnose it. Con: the sweep figure has an unexplained kink if nobody reads
  the text.

**Decision:** The third. `experiments/example_parameter_sweep.py` prints an
explanation of the artefact and advises plotting the difficulty distribution
whenever a result looks surprising.

**Impact:** Results with spread greater than roughly a third of the mean should
be treated with suspicion. Plot the thresholds before believing them.

---

## Issue 8: Tests must run without pytest

**Status:** Resolved

**Context:** pytest could not be installed in the development environment (no
network). More importantly, an assignment that fails at `pip install` before a
student has run a single line is a bad assignment.

**Options considered:**
- Require pytest. Pro: standard, better output. Con: one more thing to fail.
- **Write pytest-compatible test functions plus a small dependency-free runner
  (`run_tests.py`) that discovers and executes them.** Pro: works either way;
  `pytest` still works for anyone who has it. Con: about 60 lines of runner to
  maintain, and no fixtures or parametrisation.

**Decision:** The second. Tests are plain functions using bare `assert`, which
is also the easiest form for students to read and extend.

**Impact:** No pytest-specific features anywhere in `tests/`. If someone adds a
fixture, it will work under pytest and break under `run_tests.py`.

---

## Issue 9: Explicit time loop rather than full vectorisation

**Status:** Resolved

**Context:** The whole simulation could be one line: count how many thresholds
fall below each time point. That is faster and shorter, but it deletes the
mechanism from the code, and the mechanism is what students are meant to
understand.

**Decision:** Keep an explicit `for` loop over time steps, vectorise across words
inside it with a comment giving the loop-over-words equivalent at each such line.
The one-line version appears in the test suite instead
(`test_curve_matches_counting_thresholds_directly`), where it serves as an
independent check on the simulation.

**Impact:** The full run takes about five seconds for 10,000 words over 7,000
steps. The redundancy between the two formulations is a feature: they must agree
exactly, and a test enforces it.

---

## Issue 10: No classes anywhere

**Status:** Resolved

**Context:** Requested constraint: readable for someone new to Python.

**Decision:** No `class` statements at all, including dataclasses and
NamedTuples. Simulations return plain dictionaries with documented keys. Both
simulation functions return the same key names so downstream code does not need
to know which model produced a result.

**Impact:** Dictionary keys are checked at runtime, not by an editor, so a typo
in a key name surfaces as a `KeyError` during a run rather than as a warning
while typing. This is the main cost of the constraint, and it is what caused the
one integration failure during development (`run_all.py` still referenced
`first_quarter` after `describe_growth` changed its return keys, caught by the
end-to-end test).

---

## Issue 11: The Wordbank comparison is optional and shape-only

**Status:** Resolved

**Context:** Comparing the model to real vocabulary norms is the most tempting
figure in the project and the easiest to over-read.

**Decision:** The figure is only produced if the user supplies `wordbank.csv`;
otherwise the run prints that it is being skipped. Both curves are rescaled to
0-to-1 on both axes.

**Impact:** A visual shape match between the model and real norms is weak
evidence. Many mechanisms produce sigmoid growth. The figure is there to prompt
the question "what would distinguish these?", not to settle it.

---

*Entries below this line are added as further decisions arise.*

---

## Issue 12: Which MRC file to use, and why the obvious one is wrong

**Status:** Resolved

**Context:** The MRC Psycholinguistic Database circulates in several forms and
it is easy to pick the wrong one. The repository originally suggested for this
project (`martingrzzler/mrc-psycholinguistic-database`) contains the 1987
distribution: `mrc2.dct` is a fixed-width flat file with no header, shipped
alongside C programs written to parse it. That repository also contains a
`dataset.tsv` which looks convenient but is not: no header row, two columns,
9,240 lines, and the numeric column is a ratings scale rather than frequency.
Its ranking gives `banana` 644 and `alligator` 627 against `and` 226 and
`about` 225, which is the signature of imageability or concreteness, not
frequency.

**Options considered:**
- Parse `mrc2.dct` directly. Pro: canonical source. Con: needs a fixed-width
  parser driven by the field table in `mrc2.doc`, which is a chunk of work and
  a chunk of things to get subtly wrong.
- Use `dataset.tsv`. Rejected: it does not contain frequencies at all.
- **Use a pre-parsed CSV of the full database.** Pro: has a header, loads
  directly, no parsing step for students. Con: third-party re-hosting, so the
  parse is someone else's work and has to be sanity-checked.

**Decision:** Use the Hugging Face dataset
`StephanAkkerman/MRC-psycholinguistic-database`
(`mrc_psycholinguistic_database.csv`, ~10 MB, MIT licence, sourced from the UWA
site). Its header was verified against the loader before adopting it.

**A trap worth recording:** the real file has three adjacent frequency columns
-- `KF Written Frequency`, `Thorndike-Lorge Frequency`, `Brown Verbal
Frequency`. The loader's original substring matching found the right one only
because it happened to come first in column order. That is luck, not logic, so
`kf_written_frequency` is now the first entry in the preference list and
`test_mrc_loader_picks_kf_column_from_the_real_header` uses the genuine header
with values arranged so that reading the wrong column produces a different
word ranking. If that test ever fails, the loader is reading the wrong data.

**Impact:** `data/README.md` now gives a direct download command and tells
readers what a wrong-column load looks like (concrete nouns at the top of the
list instead of function words). The Kucera-Francis limitation from Issue 5 is
unchanged: this is 1960s written American English either way.
