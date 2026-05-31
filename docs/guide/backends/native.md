# Native (PyTorch Lightning)

[`AutonmtTranslator`](../../reference/backends.md) is the default backend and AutoNMT's own
engine. It wraps one of your [models](../models/using-a-model.md) with PyTorch Lightning and
owns the DataLoaders, callbacks (early stopping, checkpointing), loggers (TensorBoard / W&B),
checkpoint loading, and the decoding step — so the two verbs `fit` and `predict` are all you
call.

It's the only backend where **you** control the architecture and the decoding algorithm, so
the rest of the User guide is effectively its manual:

- the architectures and how to write your own → [Models](../models/using-a-model.md)
- training knobs, schedules, bucketing → [Training](../training/training.md)
- decoding strategies and `predict` → [Translation](../translation/generating.md)

This page just covers what's specific to the *translator object* itself.

## Construct it

```python
from autonmt.backends import AutonmtTranslator
from autonmt.core.nn.models import Transformer

src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)

trainer = AutonmtTranslator.from_dataset(
    train_ds,
    model=Transformer.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    run_prefix="exp",
)
```

### `from_dataset` vs the plain constructor

`from_dataset(train_ds, run_prefix=..., **kwargs)` is the convenient path: it resolves the
on-disk **run location** (`models/autonmt/runs/<run_name>/`) and a `run_name` from the
dataset variant, so checkpoints, logs, and translations land in the right place
automatically. It's equivalent to:

```python
AutonmtTranslator(
    model=...,
    src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    runs_dir=train_ds.get_runs_path(toolkit="autonmt"),
    run_name=train_ds.get_run_name(run_prefix="exp"),
)
```

Calling the constructor directly is the manual path — useful when you want a custom
`run_name` (`"ablation-v2-seed42"`) or a different directory scheme. See
[How-to → Drive the pipeline manually](../../how-to/manual-pipeline.md).

## The experiment loop

`fit` then `predict` is the whole cycle:

```python
from autonmt.backends._base.config import FitConfig, PredictConfig

trainer.fit(train_ds, config=FitConfig(max_epochs=10, batch_size=128))
scores = trainer.predict(test_variants, config=PredictConfig(beams=[5], metrics={"bleu", "chrf"}))
```

- [`fit`](../training/training.md) builds train/val DataLoaders, configures the
  optimizer/scheduler/criterion, attaches callbacks and loggers, and saves checkpoints.
- [`predict`](../translation/generating.md) decodes the test set(s) with a
  [search strategy](../translation/decoding.md), scores with the requested
  [metrics](../evaluation/metrics.md), and returns score dicts for the
  [report](../evaluation/reports.md).

---

Prefer to start from a pretrained checkpoint instead of training from scratch? See
**[HuggingFace](huggingface.md)**.
