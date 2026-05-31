# Understanding the output

The [quickstart](quickstart.md) printed a score table — but its more important product is
what it wrote to disk. AutoNMT's whole value proposition is that an experiment leaves
behind **reproducible, self-describing artifacts**, organized the same way every time. This
page is a guided tour of what just got created and where to look for each thing.

After the run, you have two trees:

```
data/multi30k/de-en/original/         # the dataset cell — reusable across runs
└── models/autonmt/runs/first_bpe_4000/   # one run inside that cell
outputs/first/reports/                # the report Report.save() wrote
```

The key idea to notice up front: **the dataset cell is independent of the run.** Prepared
data is computed once and shared; each training run nests *inside* its dataset cell. Swap
the model or the seed and you get a new run folder — the encoded data and vocabularies are
not rebuilt.

## The dataset cell

```
data/multi30k/de-en/original/
├── data/
│   ├── 0_raw/                  # the corpus as it arrived
│   ├── 1_splits/               # train.de / train.en / val.* / test.*
│   ├── 2_preprocessed/         # cleaned (normalized, filtered, deduped)
│   └── 4_encoded/bpe/4000/     # subword-encoded splits for this (model, vocab)
├── vocabs/bpe/4000/            # the trained tokenizer + vocabulary files
└── stats/bpe/4000/             # length / token-count statistics
```

The folders are **numbered by stage**. Each stage checks whether its output already exists
before doing any work, so re-running the script recomputes nothing that's already on disk
(see `force_overwrite` in [Configuration & reproducibility](../guide/experiments/configuration.md)).
The filename extension is always the language code — `train.de`, `train.en` — never a
subdirectory.

!!! tip "Debugging a single stage"
    If a stage looks wrong, delete **only that stage's folder** (e.g. `4_encoded/bpe/4000/`)
    and re-run. AutoNMT rebuilds just the missing stage and everything downstream of it,
    leaving the rest untouched.

## The run

```
models/autonmt/runs/first_bpe_4000/
├── checkpoints/                # best.ckpt / last.ckpt
├── logs/
│   ├── config_train.json       # the full effective FitConfig used
│   ├── config_predict.json     # the full effective PredictConfig used
│   └── …                       # TensorBoard event files
└── eval/<eval_ds>/
    └── translations/beam5/
        ├── src.txt             # source sentences
        ├── ref.txt             # gold references
        ├── hyp.txt             # the model's translations
        └── scores/             # one file per metric backend (bleu, chrf, …)
```

- **`checkpoints/`** — the best (by validation metric) and last weights. `predict()` loads
  `best` by default.
- **`logs/`** — TensorBoard event files (open with `tensorboard --logdir`) **and** the two
  config dumps. Those JSON files are the run's memory: the exact configuration it ran with,
  plus an environment snapshot (Python and package versions, the git SHA, and whether the
  tree was dirty). Months later you can read them and know precisely what produced a number.
- **`eval/<eval_ds>/translations/beam5/`** — the decoded output, one folder per beam width.
  `src`/`ref`/`hyp` are the parallel triple you can eyeball; `scores/` holds the raw metric
  artifacts that the report reads back.

!!! info "Why a beam-width folder?"
    `predict(config=PredictConfig(beams=[1, 5]))` decodes the same checkpoint twice and
    writes `beam1/` and `beam5/` side by side, so you can compare decoding settings without
    re-training. The beam width is part of the path, so nothing overwrites.

## The report

```
outputs/first/reports/
├── report.json                 # every score, fully nested
├── report.csv                  # the same, flattened to one row per run × eval set
└── report_summary.csv          # the compact table print(report) shows
```

`report.csv` flattens nested scores into columns named `<tool>_<metric>_<field>` — e.g.
`sacrebleu_bleu_score`, `sacrebleu_chrf_score` — under each `translations.beam<N>` block.
That flat schema is what the [reporting](../guide/evaluation/reports.md) plots and tables
consume, and it's stable across runs so you can diff or concatenate reports.

## What makes this reproducible

Three properties, all visible in the trees above:

1. **Path-driven, staged data.** Every artifact has one computed location; stages skip when
   present, so re-runs are incremental.
2. **Self-describing runs.** Each run dumps its full config plus an environment/git snapshot
   next to its outputs.
3. **A single seed.** `FitConfig(seed=...)` seeds Python, NumPy, PyTorch, and Lightning
   together.

The full canonical layout — including the deeper `eval/.../data/` staging and the
multi-toolkit `models/<toolkit>/` split — is documented in
[Concepts → On-disk layout](../concepts/on-disk-layout.md), and the reproducibility
machinery in [Concepts → Reproducibility model](../concepts/reproducibility.md).

---

You've now seen a full experiment and its artifacts. The [User guide](../guide/experiments/workflow.md)
picks up from here: how to turn this one-cell run into a real grid, and how to configure
each block of the pipeline.
