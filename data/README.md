# Optional data files

**Nothing in this project requires these files.** If the folder is empty,
everything still runs using synthetic word frequencies, and every figure built
that way is labelled `SYNTHETIC` so it cannot be mistaken for real data.

Adding real data lets you check whether conclusions drawn from an invented
vocabulary survive contact with a real one. That is worth doing, and it is a
good exercise, but it is not required for any of the core results.

---

## 1. Word frequencies: the MRC Psycholinguistic Database

**What it adds:** replaces the synthetic adult frequency list with real
frequency counts for English words, so the frequency-based model runs on a real
vocabulary.

**Where to get it:** a pre-parsed CSV of the full database is hosted on Hugging
Face under an MIT licence. No account or API token is needed:

```bash
cd data
curl -L -o mrc.csv \
  "https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database/resolve/main/mrc_psycholinguistic_database.csv?download=true"
```

That is about 10 MB and roughly 150,000 rows.

**Avoid the raw distribution unless you enjoy pain.** The original MRC release
(`mrc2.dct`) is a fixed-width flat file with no header, which is why it ships
with C programs to read it. Some repositories also publish a two-column
`dataset.tsv` containing only a ratings column, with no frequencies at all --
if the top of your word list looks like `banana` and `alligator` rather than
`the` and `of`, that is what you have.

**Format expected:** a CSV with a header row containing a column named `Word`
and one named `KF Written Frequency`. The file above has both. The loader also
accepts several near-miss spellings, and it deliberately prefers the
Kucera-Francis column over the Thorndike-Lorge and Brown verbal frequency
columns that sit next to it in the same file.

```
Word,Number of Letters,Number of Phonemes,...,KF Written Frequency,...
THE,3,1,1,...,69971,...
DOG,3,3,1,...,75,...
```

Check it loaded, and sanity-check the ranking:

```bash
python -c "import datasets; w,f = datasets.load_mrc_frequencies('data/mrc.csv', 2000); print(len(w)); print(w[:15])"
```

You should see ordinary function words at the top: *the, of, and, to, a*. If you
see concrete nouns instead, the loader has found the wrong column; send the
header line to whoever is maintaining this and pin the column explicitly.

Entries with zero frequency are skipped, repeated senses of the same word are
collapsed to one entry, and non-words such as `&ARRY` are filtered out.

**Worth knowing:** MRC frequency counts come from Kucera-Francis, which is
written American English from the 1960s, not speech to children. That is a real
limitation, not a detail: the words a toddler hears most are not the words that
appear most in 1960s print. Treat the MRC run as "difficulty built from a real
word-frequency distribution", not as "difficulty built from what children
actually hear".

**Other columns you might use.** The file also carries number of phonemes,
number of syllables, familiarity, concreteness, imageability, and an age of
acquisition rating. The model builds difficulty from frequency alone, but the
idea that difficulty is the sum of many independent factors is exactly what
`make_multifactor_thresholds` illustrates with invented factors. Building it
from these real columns instead is a natural extension.

---

## 2. Child vocabulary norms: Wordbank

**What it adds:** real vocabulary growth curves from children, so you can
compare the shape the model produces against the shape children actually show.
Without this file, the model-versus-reality figure is skipped.

**Where to get it:** https://langcog.github.io/wordbank-datapage/data.html

**What to do:** download an English (American) administration-level export and
save it here as `wordbank.csv`.

**Format expected:** a CSV with a header row containing a column whose name
includes `age` (in months) and one whose name includes `production`, `vocab` or
`words`. One row per child. The loader averages within each age.

```
age,production
16,12
16,31
18,54
```

Check it loaded with:

```bash
python -c "import datasets; n = datasets.load_wordbank_norms('data/wordbank.csv'); print(n['ages'], n['mean_words_produced'])"
```

**Worth knowing:** the comparison figure rescales both curves to run from 0 to 1
on both axes, because model time steps are not months and there is no principled
way to convert between them. What is being compared is the *shape* of the two
curves, not their values. Any claim stronger than "these have a similar shape"
would need a calibration argument the model does not currently make.
