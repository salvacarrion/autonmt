# On-disk layout

AutoNMT is **path-driven**: nothing is held only in memory between stages. A `Dataset`
*computes* where every artifact belongs, each stage writes to a fixed, numbered folder, and
re-running an experiment reconstructs only what's missing. That single design choice is what
makes AutoNMT [reproducible by construction](reproducibility.md) — this page is the complete
map.

## The directory tree

Everything for one dataset cell lives under `base_path/<dataset>/<lang-pair>/<size>/`:

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
builder.build(force_overwrite=False)   # default: keep existing artifacts
builder.build(force_overwrite=True)    # rebuild everything from scratch
```

!!! warning "Debugging a stage? Delete only that stage."
    Because each stage is independent on disk, the right way to force a rebuild of (say) the
    encoding is to remove `4_encoded/<subword>/<vocab>/` and re-run — **not** to nuke the
    whole dataset tree. Checkpoints get the same care: an existing `.pt` is renamed to
    `.pt.bak` rather than clobbered when `force_overwrite=True`.

## Backends share the layout

The only part of the tree that varies by backend is the `<toolkit>` folder name —
`models/autonmt/…`, `models/huggingface/…`, `models/fairseq/…`. Everything else (the data
stages, the `eval/.../beam<N>/` artifacts, the score files) is identical, which is what lets
a single [report](../guide/evaluation/reports.md) compare runs across backends.

---

Where the *layout* ends, the **[Reproducibility model](reproducibility.md)** begins: how each
run records what produced it.
