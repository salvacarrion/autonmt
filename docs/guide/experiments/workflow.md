# The experiment workflow

Every AutoNMT experiment has the same shape, no matter how big it gets:

> **declare a grid → unroll it into dataset variants → run each through a translator →
> collect the scores into one report.**

This page is the practical version of that idea — how you structure it in code. For the
picture behind it, see [Concepts → The mental model](../../concepts/mental-model.md).

## 1 · Declare the grid

You don't loop over datasets and hyper-parameters; you **declare the axes** and let the
[`DatasetBuilder`](../data/datasets.md) own the cross-product.

```python
from autonmt.datasets import DatasetBuilder

builder = DatasetBuilder(
    base_path="data",
    datasets=[
        {"name": "multi30k", "languages": ["de-en", "fr-en"],
         "sizes": [("original", None), ("50k", 50000)]},
    ],
    encoding=[
        {"subword_models": ["bpe", "unigram"], "vocab_sizes": [4000, 8000]},
    ],
).build()
```

That declaration describes **2 language pairs × 2 sizes × 2 subword models × 2 vocab sizes
= 16** dataset variants. `.build()` runs the data pipeline for each cell (clean → split →
learn tokenizer → encode → build vocab), writing everything to disk and skipping any stage
already present.

## 2 · Unroll into variants

The builder hands back the variants as flat lists — one [`Dataset`](../data/datasets.md#the-dataset-object)
per cell:

```python
train_variants = builder.get_train_ds()   # list[Dataset], one per cell
test_variants  = builder.get_test_ds()
```

A `Dataset` is not a PyTorch dataset — it's an **identity plus a path engine**. Given
*(name, language pair, size, subword model, vocab size)* it knows where every file for that
cell lives, so you never compute paths yourself.

## 3 · Bind a translator and run the two verbs

A [translator](../backends/choosing.md) turns a variant into a trained model and then into
scored translations. It exposes exactly two verbs — `fit` and `predict` — and which
translator you instantiate decides *which toolkit* runs underneath (native, HuggingFace,
Fairseq) without changing those verbs.

```python
from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.nn.models import Transformer

train_ds = train_variants[0]
src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)

trainer = AutonmtTranslator.from_dataset(
    train_ds,
    model=Transformer.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    run_prefix="sweep",
)

trainer.fit(train_ds, config=FitConfig(max_epochs=10, batch_size=128))
scores = trainer.predict(test_variants, config=PredictConfig(beams=[5], metrics={"bleu", "chrf"}))
```

!!! info "`from_dataset` resolves the run location for you"
    `from_dataset(train_ds, run_prefix="sweep", ...)` derives the on-disk run directory
    (`models/autonmt/runs/<run_name>/`) and a `run_name` from the variant, so checkpoints,
    logs, and translations land in the right place automatically. Need a custom run name or
    directory scheme? Call the constructor directly — see
    [How-to → Drive the pipeline manually](../../how-to/manual-pipeline.md).

- [**`fit`**](../training/training.md) trains the model: DataLoaders, optimizer/scheduler,
  callbacks, loggers, checkpoints.
- [**`predict`**](../translation/generating.md) decodes the test set(s) with a
  [search strategy](../translation/decoding.md), scores the output with the requested
  [metrics](../evaluation/metrics.md), and returns score dicts.

## 4 · The whole experiment is a flat loop

Because the builder already unrolled the grid, your experiment is **iteration, not
nesting** — the loop body is identical no matter how many axes you swept:

```python
from autonmt.reporting.report import Report

scores = []
for train_ds in builder.get_train_ds():                 # one cell of the grid
    src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)
    trainer = AutonmtTranslator.from_dataset(
        train_ds,
        model=Transformer.from_vocabs(src_vocab, tgt_vocab),
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        run_prefix="sweep",
    )
    trainer.fit(train_ds, config=FitConfig(max_epochs=10))
    scores.append(trainer.predict(builder.get_test_ds(),
                                  config=PredictConfig(metrics={"bleu", "chrf"})))

(
    Report.from_runs(scores, output_path="outputs/sweep")
    .save()
    .plot_comparison("bleu", beam=5)
)
```

The complexity lives in the **declaration**, not the control flow. That's what makes the
final table a controlled comparison rather than a pile of runs: every cell flowed through
the same code, so the only thing that differs is the axis you swept.

## Where each block is documented

The rest of the User guide opens each block of that loop:

| Block | Page |
| --- | --- |
| Prepare data, tokenize, build vocab | [Data](../data/datasets.md) |
| Pick or write a model | [Models](../models/using-a-model.md) |
| Train (`fit`) | [Training](../training/training.md) |
| Translate & decode (`predict`) | [Translation](../translation/generating.md) |
| Score & report | [Evaluation](../evaluation/metrics.md) |
| Run it on another toolkit | [Backends](../backends/choosing.md) |

How configs are merged, persisted, and seeded — the reproducibility side of the loop — is
the next page: [Configuration & reproducibility](configuration.md).
