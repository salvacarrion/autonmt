# Preprocessing

Between "raw parallel text" and "tensors a model can train on" sit two distinct jobs, and
AutoNMT keeps them deliberately separate:

- **Preprocessing** *(this page)* — *subword-agnostic* text cleanup: normalize, filter by
  length, dedupe, shuffle. The output is the same whether you'll encode with `word`, `bpe`,
  or `bytes`.
- **[Encoding](tokenization.md)** — *subword-dependent* tokenization: turn clean text into
  subword units with a learned (or rule-based) tokenizer.

The split matters because it's why a grid over `subword_models × vocab_sizes` is cheap:
preprocessing runs **once** per split (`2_preprocessed/`), and only the encoding stage
(`4_encoded/<sw>/<vs>/`) re-runs per subword/vocab combination.

## The `2_preprocessed` stage

`autonmt.datasets.preprocessing` provides composable cleanup functions. You usually reach
them through the builder's [hooks](#hooks); the building blocks are:

| Function | What it does |
| --- | --- |
| `normalize_lines(lines, seq=...)` | Unicode normalization (default `NFKC` + `Strip`) via `tokenizers.normalizers` |
| `preprocess_pairs(src, tgt, ...)` | Paired cleanup: normalize, length filter, length-ratio filter, dedupe, shuffle — keeping src/tgt aligned |
| `preprocess_lines(lines, ...)` | Single-side cleanup (used at predict time on the source only) |

`preprocess_pairs` is the workhorse for training data. Its knobs:

```python
preprocess_pairs(
    src_lines, tgt_lines,
    normalize_fn=normalize,        # e.g. NFKC + Strip
    min_len=1, max_len=None,       # drop empty / overly long sentences
    max_len_percentile=None,       # or cap length at a percentile of the corpus
    remove_duplicates=True,        # drop exact duplicate pairs
    max_len_ratio_percentile=99,   # drop pairs whose src/tgt lengths differ wildly
    safe_len_ratio=2.0,            # …but never below this ratio threshold
    shuffle_lines=False,
)
```

!!! info "Why filter by length ratio?"
    A source sentence of 5 tokens aligned to a target of 60 is almost always a misalignment
    or a bad scrape, not a real translation. Such pairs add noise and waste capacity. The
    ratio filter drops pairs where `max(len)/min(len)` exceeds a data-driven threshold (the
    `max_len_ratio_percentile`), but never goes below `safe_len_ratio` so legitimate
    variation survives.

## Hooks: injecting your own cleanup { #hooks }

You don't subclass anything to customize preprocessing — you pass **callable hooks** to the
builder. There are three, fired at different stages:

| Hook | Fired on | Signature |
| --- | --- | --- |
| `preprocess_raw_fn` | the raw files, before splitting | `fn(data, ds) -> (src_lines, tgt_lines)` |
| `preprocess_splits_fn` | each split, after splitting | `fn(data, ds) -> (src_lines, tgt_lines)` |
| `preprocess_fn` *(predict-time)* | the source at translation time | `fn(data, ds) -> lines` |

For the paired hooks, `data` carries `data["src"]["lines"]` and `data["tgt"]["lines"]`; for
the predict-time hook, `data["lines"]` and `data["lang"]`. A typical setup wires the same
normalization through all of them:

```python
from tokenizers.normalizers import NFKC, Strip
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.preprocessing import normalize_lines, preprocess_pairs, preprocess_lines

def normalize(lines):
    return normalize_lines(lines, seq=[NFKC(), Strip()])

def clean_pairs(data, ds):                 # for preprocess_raw_fn / preprocess_splits_fn
    return preprocess_pairs(
        data["src"]["lines"], data["tgt"]["lines"],
        normalize_fn=normalize, remove_duplicates=True, max_len_ratio_percentile=99,
    )

def clean_source(data, ds):                # for the predict-time preprocess_fn
    return preprocess_lines(data["lines"], normalize_fn=normalize)

builder = DatasetBuilder(
    base_path="data", datasets=[...], encoding=[...],
    preprocess_raw_fn=clean_pairs,
    preprocess_splits_fn=clean_pairs,
).build()
```

The predict-time `preprocess_fn` is passed to [`predict`](../translation/generating.md)
instead, so your test source gets the *same* normalization the training data saw — otherwise
you'd measure a train/test preprocessing mismatch rather than model quality.

!!! warning "Match train and predict preprocessing"
    Whatever normalization you apply to training data, apply the equivalent to the
    predict-time source via `preprocess_fn`. A capitalization or Unicode-form mismatch between
    the two silently depresses scores.

---

Clean text in hand, the next stage turns it into subword units:
**[Subword tokenization](tokenization.md)**.
