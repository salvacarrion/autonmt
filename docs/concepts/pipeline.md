# The pipeline

Every AutoNMT experiment is three composable layers. Once you see them, every example
script reads the same way.

```{.text .pipeline}
DatasetBuilder  ───►  Translator  ───►  generate_report
   (the grid)        (fit / predict)     (json + csv + plots)
                          │
          ┌───────────────┼────────────────────┐
          ▼               ▼                     ▼
  AutonmtTranslator  HuggingFaceTranslator  FairseqTranslator
  (Lightning models)  (transformers seq2seq)  (fairseq CLI, deprecated)
```

## Layer 1 — `DatasetBuilder`: the grid

[`DatasetBuilder`][autonmt.datasets.dataset_builder.DatasetBuilder] takes your declared
axes and unrolls the cross-product of `datasets × language pairs × sizes × subword models
× vocab sizes`. For each cell it:

1. runs your cleanup hooks (`preprocess_raw_fn`, `preprocess_splits_fn`),
2. trains a SentencePiece tokenizer,
3. encodes the splits,

and writes everything to disk. `get_train_ds()` / `get_test_ds()` return the lists of
[`Dataset`][autonmt.datasets.dataset.Dataset] objects the experiment loop iterates over.

A `Dataset` here is a **path/identity holder** — it knows where its files live and what
its variant is. It is *not* a PyTorch `Dataset` (that's
[`TranslationDataset`][autonmt.core.data.translation_dataset.TranslationDataset], used
internally by the AutoNMT backend).

→ [The grid](grid.md) · [Preprocessing & vocabularies](../guides/preprocessing-and-vocabs.md)

## Layer 2 — `Translator`: fit & predict

A translator wraps a model (or an external toolkit) and exposes two methods:

- **`fit(train_ds, config=FitConfig(...))`** trains on a dataset cell.
- **`predict(test_datasets, config=PredictConfig(...))`** translates the test set(s) and
  scores the hypotheses, returning a per-run score dict.

All backends inherit this contract from
[`BaseTranslator`][autonmt.backends._base.translation_engine.BaseTranslator]. The concrete
class you pick decides the engine:

| Translator | Engine | Use it for |
| --- | --- | --- |
| [`AutonmtTranslator`](../backends/autonmt.md) | PyTorch Lightning | Training AutoNMT's own models (default) |
| [`HuggingFaceTranslator`](../backends/huggingface.md) | `transformers` | Evaluating / fine-tuning pretrained seq2seq checkpoints |
| [`FairseqTranslator`](../backends/fairseq.md) | Fairseq CLI | Legacy flows (**deprecated**) |

Because the surface is identical, swapping engines means changing the class and little
else.

### Config: kwargs or typed objects

`fit()` and `predict()` accept either keyword arguments or a typed
[`FitConfig`][autonmt.backends._base.config.FitConfig] /
[`PredictConfig`][autonmt.backends._base.config.PredictConfig] object:

```python
from autonmt.backends._base.config import FitConfig

# Equivalent
trainer.fit(train_ds, batch_size=64, max_epochs=10)
trainer.fit(train_ds, config=FitConfig(batch_size=64, max_epochs=10))

# Explicit kwargs override the config, per key
trainer.fit(train_ds, config=FitConfig(batch_size=64), max_epochs=20)  # → 20 epochs
```

Toolkit-specific extras (`wandb_params`, `fairseq_args`, `strategy`, …) pass through
`**kwargs` untouched and are forwarded to the underlying backend.

## Layer 3 — `generate_report`: one comparable table

[`generate_report`][autonmt.reporting.report.generate_report] takes the list of score
dicts your `predict()` calls returned and flattens them into a single CSV + JSON plus
comparison plots. Score keys are flattened as
`translations.beam<n>.<tool>_<metric>_<field>`, e.g.
`translations.beam5.sacrebleu_bleu_score`.

```python
df_report, df_summary = generate_report(
    scores=scores, output_path="outputs/run1",
    plot_metric="translations.beam5.sacrebleu_bleu_score",
)
```

→ [Reports & plots](../guides/reports.md)

---

The rest of the docs drill into each layer. If you want to see all three with the
abstractions peeled away, read [Under the hood](../guides/under-the-hood.md).
