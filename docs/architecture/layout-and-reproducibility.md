# On-disk layout & reproducibility

AutoNMT is **path-driven**: nothing is held only in memory between stages. A `Dataset`
*computes* where every artifact belongs, each stage writes to a fixed, numbered folder, and
re-running an experiment reconstructs only what's missing. That single design choice is what
makes AutoNMT [reproducible by construction](../introduction/philosophy.md#reproducible) —
this page shows the layout and the mechanisms that rely on it.

## The directory tree

Everything for one dataset cell lives under
`base_path/<dataset>/<lang-pair>/<size>/`:

```text
data/multi30k/de-en/original/
├── data/
│   ├── 0_raw/                     # raw files (data.de / data.en)
│   ├── 1_raw_preprocessed/        # optional cleanup of raw files
│   ├── 1_splits/                  # train/val/test splits  (train.de, train.en, ...)
│   ├── 2_preprocessed/            # per-split cleanup (normalize, filter, dedupe)
│   ├── 3_pretokenized/            # Moses pretokenization (only when subword=word)
│   └── 4_encoded/<subword>/<vocab>/   # subword-encoded files  (e.g. bpe/4000/)
├── vocabs/<subword>/<vocab>/      # SentencePiece .model/.vocab + frequencies (.vocabf)
├── stats/<subword>/<vocab>/       # per-split token statistics
├── plots/<subword>/<vocab>/       # dataset diagnostic figures
└── models/<toolkit>/runs/<run_name>/
    ├── checkpoints/               # best / last .pt
    ├── logs/                      # TensorBoard + config_train.json / config_predict.json
    └── eval/<eval_ds>/
        ├── data/{0_raw, 1_preprocessed, 3_encoded}
        └── translations/beam<N>/  # src.txt, ref.txt, hyp.txt
            └── scores/             # <backend>_scores.{json,txt}
```

Two things are worth internalizing:

1. **The numbers encode the order of operations.** `0_raw` → `1_splits` →
   `2_preprocessed` → `3_pretokenized` → `4_encoded`. You can read the pipeline straight off
   the filesystem.
2. **Data and runs are separate subtrees.** `data/`, `vocabs/`, `stats/` describe the
   *dataset cell* and are reused by every run; `models/<toolkit>/runs/<run_name>/` describes
   one *training run*. Several models can share one prepared dataset, and each run is
   self-contained.

!!! info "Subword & vocab are part of the path"
    Notice `4_encoded/bpe/4000/` and `vocabs/bpe/4000/`. The subword model and vocab size are
    *directories*, so the same split can be encoded multiple ways side by side without
    collision — exactly what a grid over `subword_models × vocab_sizes` needs. (For `word`
    runs, encoding *is* Moses pretokenization, so the `3_pretokenized/` stage doubles as the
    encoded source.)

## The naming convention

Filenames use the **language code as the extension**, never as a directory:

```text
train.de   train.en        ✅  (de = source, en = target)
de/train.txt               ❌  not how AutoNMT works
```

This is consistent everywhere — splits, encoded files, and the `src.txt`/`ref.txt`/`hyp.txt`
evaluation artifacts. A `Dataset` derives the extensions from its `lang_pair`, so you rarely
type them yourself.

## Stage skipping: why re-runs are cheap

Every stage checks for its output before doing work, gated by a `force_overwrite` flag.
Run the same script twice and the second run **skips** preprocessing, encoding, vocab
building, and even training if a checkpoint already exists — it jumps straight to whatever
is missing.

```python
# Re-prepare a single stage by deleting only its folder, not the whole tree:
#   rm -r data/multi30k/de-en/original/data/4_encoded/bpe/4000
# …then re-run. AutoNMT rebuilds just the encoded files (and downstream).

builder.build(force_overwrite=False)   # default: keep existing artifacts
builder.build(force_overwrite=True)    # rebuild everything from scratch
```

!!! warning "Debugging a stage? Delete only that stage."
    Because each stage is independent on disk, the right way to force a rebuild of (say) the
    encoding is to remove `4_encoded/<subword>/<vocab>/` and re-run — **not** to nuke the
    whole dataset tree. Checkpoints get the same care: `_train` renames an existing `.pt` to
    `.pt.bak` rather than clobbering it when `force_overwrite=True`.

## Every run records itself

Reproducibility isn't only about *files in the right place* — it's about knowing **what
produced them**. On every `fit()` / `predict()`, `BaseTranslator`:

- **Dumps the effective config** to `logs/config_train.json` / `config_predict.json` — the
  fully-merged settings (defaults < `config=` < kwargs), so there's no ambiguity about which
  values actually ran.
- **Snapshots the environment** into that config: Python version, the versions of `torch`,
  `pytorch_lightning`, `transformers`, `sentencepiece`, `sacrebleu`, `autonmt`, plus the
  working directory, `argv`, and — when the cwd is a git repo — the **commit SHA and a dirty
  flag**.

```json
{
  "environment": {
    "python": "3.12.4", "torch": "2.4.0", "autonmt": "1.0.0",
    "git_sha": "26436ac…", "git_dirty": false
  },
  "fit": { "max_epochs": 5, "batch_size": 128, "learning_rate": 0.001, ... }
}
```

!!! note "Configs keep primitives only"
    The config dump renders non-primitive values readably (a callable hook becomes
    `module.qualname`) but doesn't try to serialize arbitrary objects. So you'll always see
    *which* `preprocess_fn` or `decoder` a run used, but the object itself won't round-trip.

## One seed for everything

```python
trainer.fit(train_ds, config=FitConfig(seed=42))
```

Passing `seed=` makes `fit` call `manual_seed(42)`, which seeds Python's `random`, NumPy,
PyTorch, and Lightning together. For full determinism *before* `fit` (vocab construction,
custom data prep), call it yourself up front:

```python
from autonmt.utils.seed import manual_seed
manual_seed(seed=42)
```

!!! tip "Seeds don't make neural training bit-identical"
    A single seed removes the *avoidable* variance (data order, init), but GPU non-determinism
    and hardware differences remain. For publication-grade comparisons, run multiple seeds and
    report mean ± std — and use a significance test for close results. AutoNMT deliberately
    leaves multi-seed as a `for` loop in *your* code (each seed gets its own `run_name`); see
    [Full manual control](../toolkit/manual-control.md#multi-seed) and
    [significance testing](../evaluation/metrics.md#significance).

---

That completes the architecture tour. From here, follow the pipeline:

- **Input side** → [Data & vocabularies](../data/dataset-builder.md)
- **The engine** → [The AutoNMT toolkit](../toolkit/overview.md)
- **Output side** → [Evaluation & reports](../evaluation/metrics.md)
