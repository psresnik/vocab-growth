# Running this project in Google Colab

This guide is for running the `vocab-growth` code in
[Google Colab](https://colab.research.google.com/) instead of on your own
computer. Colab gives you a free notebook environment in your browser with
Python already installed, which sidesteps a lot of local setup problems —
but it works a little differently from a normal computer, so this guide is
longer than it would otherwise need to be. Follow it top to bottom the first
time; after that it's three or four lines.

If you are comfortable with the command line and already have Python working
locally, you probably don't need this guide — see the main
[README.md](README.md) instead.

---

## Contents

- [Before you start](#before-you-start)
- [Step 1: Open a new Colab notebook](#step-1-open-a-new-colab-notebook)
- [Step 2: Clone the repository](#step-2-clone-the-repository)
- [Step 3: Install the dependencies](#step-3-install-the-dependencies)
- [Step 4: Run the tests](#step-4-run-the-tests)
- [Step 5: Run the full pipeline](#step-5-run-the-full-pipeline)
- [Step 6: View the figures](#step-6-view-the-figures)
- [Step 7: Download your results](#step-7-download-your-results)
- [Optional: keep your work between sessions](#optional-keep-your-work-between-sessions)
- [Editing the code in Colab](#editing-the-code-in-colab)
- [Troubleshooting](#troubleshooting)

---

## Before you start

A few things about Colab that are worth knowing before you begin, especially
if you have never used it:

- **A Colab notebook is a sequence of cells.** Each cell holds either code or
  text. You run a code cell with **Shift+Enter**, or by clicking the little
  play button that appears on the left when you hover over the cell.
- **Lines starting with `!` run a shell command**, not Python. `!git clone
  ...` and `!pip install ...` below are shell commands, exactly as if you
  typed them at a terminal prompt. Lines *without* `!` are ordinary Python.
- **Everything you do disappears when the session ends**, unless you
  explicitly saved it somewhere else (like your Google Drive, or by
  downloading a file to your own computer). Closing the tab, or leaving Colab
  idle for a while, ends the session. This is the single most important thing
  to understand about Colab and is covered more in
  [Optional: keep your work between sessions](#optional-keep-your-work-between-sessions).
- **You do not need to install Python.** Colab notebooks already have Python,
  `numpy`, and `matplotlib` available. You will still run one `pip install`
  command below, mainly to get `pytest`, and to make sure your versions match
  what this project expects.

---

## Step 1: Open a new Colab notebook

1. Go to [colab.research.google.com](https://colab.research.google.com/).
2. Sign in with a Google account if you are not already signed in.
3. Click **New notebook**.

You should see a single empty code cell. That's it — you're ready for step 2.

---

## Step 2: Clone the repository

"Cloning" means downloading a copy of the code repository into the Colab
machine you've just been given. Click into the first cell and paste the block below.

```python
!git clone https://github.com/psresnik/vocab-growth.git
%cd vocab-growth
```

- The first line downloads the code.
- `%cd` is a special Colab/Jupyter command (called a "magic command," hence
  the `%` instead of `!`) that changes your *current directory* — think of it
  as double-clicking into the `vocab-growth` folder — for the rest of the
  notebook. Every command you run after this will run from inside that
  folder, which matters because the code expects to be run from there.

Run the cell (Shift+Enter). You should see git output ending in something
like `Resolving deltas: 100% ...`, with no red error text.


---

## Step 3: Install the dependencies

In a new cell:

```python
!pip install -r requirements.txt
```

This reads `requirements.txt` from the repository and installs anything
listed there that Colab doesn't already have (Colab comes with `numpy` and
`matplotlib` pre-installed, so this step mostly adds `pytest`). You'll see a
fair amount of output; look for a line near the end like `Successfully
installed ...` or `Requirement already satisfied` and no red error text.

You do **not** need `environment.yml` or conda in Colab — that file is for
people setting this up on their own computer with Anaconda. Colab has its
own separate Python environment that pip manages directly.

---

## Step 4: Run the tests

Before running the full simulation, confirm the code is working correctly on
this machine:

```python
!python run_tests.py
```

You should see `60 passed, 0 failed` at the end (the exact count may be
higher if more tests have been added since this guide was written — as long
as it says `0 failed`, you're fine). If you see any `FAIL` lines, see
[Troubleshooting](#troubleshooting) below before continuing.

---

## Step 5: Run the full pipeline

```python
!python run_all.py
```

This runs every simulation in the project and writes six figures, a results
table, and a log file into an `output/` folder. It takes about five seconds.
You'll see printed output describing what each part of the model is doing as
it runs.

If you just want a quick check that everything works, without waiting for
the full-size run:

```python
!python run_all.py --quick
```

See `README.md` for the other command-line options (`--seed`, `--n-words`,
`--outdir`, and so on) — they all work identically in Colab.

---

## Step 6: View the figures

`run_all.py` saves figures as image files rather than displaying them, since
that's how the code needs to work outside of Colab too. To actually see one
in your notebook, use Python's `IPython.display` module, which Colab
provides automatically:

```python
from IPython.display import Image, display

display(Image("output/fig2_baseline_growth_and_rate.png"))
```

- `Image(...)` loads the file at that path.
- `display(...)` is what actually draws it in the notebook's output area,
  below the cell.

Repeat with a different filename to see another figure — the full list is in
the "What each result is" table in the main `README.md`. If you'd rather see
all of them at once, run:

```python
import glob

for path in sorted(glob.glob("output/*.png")):
    print(path)
    display(Image(path))
```

- `glob.glob("output/*.png")` finds every file in `output/` ending in
  `.png` — the `*` is a wildcard meaning "anything." `sorted(...)` puts them
  in a predictable order (fig1, fig2, fig3, ...) rather than whatever order
  the filesystem happens to return them in.

---

## Step 7: Download your results

Since the Colab machine is temporary, anything you want to keep needs to be
downloaded to your own computer explicitly.

**To download one file** (for example, a figure you want to put in a
write-up):

```python
from google.colab import files

files.download("output/fig2_baseline_growth_and_rate.png")
```

This will trigger your browser's normal file-download prompt.

**To download the whole `output/` folder at once**, zip it first, since
browsers download one file at a time:

```python
!zip -r output.zip output
from google.colab import files
files.download("output.zip")
```

- `!zip -r output.zip output` is a shell command that bundles the entire
  `output` folder into a single file called `output.zip`. The `-r` means
  "recursive" — include everything inside the folder, not just the top
  level.

---

## Optional: keep your work between sessions

By default, every time you open a fresh Colab notebook you'll need to repeat
Steps 2 through 5 — the cloned repository and anything you installed
disappear when the session ends. For a short assignment this is usually fine
and is the simplest approach.

If you're doing more extended work and don't want to re-clone every time,
you can store the repository in your Google Drive instead, which persists.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Running this cell will prompt you to log into your Google account and
authorize access; follow the on-screen link and paste back the code it gives
you (Colab handles this automatically in most cases and just shows a
pop-up). Once mounted, your Drive is available as a regular folder at
`/content/drive/MyDrive`.

Clone into Drive instead of the temporary machine:

```python
%cd /content/drive/MyDrive
!git clone https://github.com/psresnik/vocab-growth.git   # add the token version from Step 2 if private
%cd vocab-growth
```

Next time, you can skip the clone entirely and just do:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/vocab-growth
```

then continue from Step 3 (you may need to reinstall `pytest` with `pip
install -r requirements.txt`, since installed packages do *not* persist even
when your files do — only what's inside your Drive folder is saved).

**A caution:** if you're editing code and someone else on your team is doing
the same thing in the same Drive folder, you can overwrite each other's
changes with no warning, since Drive isn't source control. If more than one
person is working on this, using `git pull` / `git push` against the GitHub
repository (rather than relying on Drive as your only copy) is safer, even
inside Colab.

---

## Editing the code in Colab

You can open and edit any file directly in the notebook interface: click the
folder icon in the left sidebar, navigate into `vocab-growth`, and
double-click a file like `config.py` to open it in an editor pane. Save with
**Ctrl+S** (or **Cmd+S** on Mac). Changes take effect the next time you run a
cell that uses that file — you do not need to re-clone or restart anything.

If you change a `.py` file that a notebook cell has already imported earlier
in the session, Python may keep using the old version it already loaded. The
reliable fix is:

```python
# Runtime menu -> Restart session
```

then re-run your cells from the top. This is a general Python/Jupyter
quirk, not specific to this project.

---

## Troubleshooting

**`fatal: repository not found` when cloning**
Either the repository is still private and you used the plain (non-token)
clone command, or your token doesn't have access. Double check which
situation you're in from [Step 2](#step-2-clone-the-repository).

**`fatal: destination path 'vocab-growth' already exists`**
You've already cloned it in this session. Either `%cd vocab-growth` directly
(skip the `git clone` line), or if you want a completely fresh copy, run
`!rm -rf vocab-growth` first.

**Tests fail, or `run_all.py` errors out with a `FileNotFoundError` or
`ModuleNotFoundError`**
Almost always means a step was skipped or run out of order. Check, in this
order:

1. Did `%cd vocab-growth` actually run, and did it print an error? Run `!pwd`
   — it should print a path ending in `/vocab-growth`.
2. Did Step 3 (`pip install`) complete without red error text?
3. Are you running commands in a *new* notebook without having re-run Steps
   2–3 in that notebook? Steps don't carry over between separate notebooks,
   only between cells *within* the same notebook and the same session.

**`ImportError: cannot import name 'files' from 'google.colab'`**
This means you're not actually running inside Colab (for example, you copied
a cell into a plain Jupyter notebook on your own computer). The
`google.colab` module in Steps 6–7 only exists inside Colab itself; on your
own machine, just use your file browser to find the files under wherever you
cloned the repository.

**Everything seems to work but you don't see any printed output**
Some cells (like the `%cd` magic command) print little or nothing when they
succeed — no output doesn't necessarily mean nothing happened. Check for red
error text specifically, rather than assuming a quiet cell has failed.

**Still stuck**
Copy the *exact* error message (the last 5-10 lines are usually the
important part) and bring it to office hours or post it on the course forum.
"It didn't work" is much harder to debug than the actual error text.
