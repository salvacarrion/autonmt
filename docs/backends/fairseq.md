# Fairseq *(deprecated)*

!!! warning "Deprecated — use the AutoNMT backend for new work"
    Fairseq was [archived by its maintainers on
    2026-03-20](https://github.com/facebookresearch/fairseq) and no longer receives updates.
    [`FairseqTranslator`](../reference/backends.md) is kept working for existing flows, but
    new projects should use the native [AutoNMT engine](../toolkit/overview.md), which gives
    you the same `fit` / `predict` surface with a maintained toolkit, in-process models, and
    [custom-architecture support](../extending/index.md#a-custom-model).

`FairseqTranslator` **shells out to the Fairseq CLI** (`fairseq-train`, `fairseq-generate`).
AutoNMT prepares the data, encodes the eval splits with the dataset's SentencePiece model
(it runs in [SPM-pipeline
mode](../architecture/toolkit-abstraction.md#spm-pipeline-mode-autonmt-fairseq)), translates
your `fit` kwargs into Fairseq flags, runs the subprocess, and parses the BLEU it reports.

Fairseq is **not** a default dependency:

```bash
pip install -e '.[fairseq]'    # deprecated
```

Importing the module emits a `DeprecationWarning`; instantiating it without `fairseq`
installed raises `ImportError` with install instructions.

## Usage

```python
from autonmt.backends import FairseqTranslator   # emits DeprecationWarning
from autonmt.backends._base.config import FitConfig

# Vocabs are still needed: the base translator encodes the eval splits with the same
# subword model the training run used.
src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)

trainer = FairseqTranslator.from_dataset(
    train_ds, src_vocab=src_vocab, tgt_vocab=tgt_vocab, run_prefix="fairseq",
)
trainer.fit(
    train_ds,
    config=FitConfig(max_epochs=5, batch_size=128),
    fairseq_args=["--arch transformer", "--dropout 0.1", "--lr 0.0005"],
)
scores = trainer.predict(test_datasets, config=PredictConfig(beams=[5], metrics={"bleu"}))
```

## The kwarg → flag bridge

AutoNMT maps common `FitConfig` fields onto Fairseq flags via an internal
`_AUTONMT_TO_FAIRSEQ` table, so the same config object works across backends:

| `FitConfig` | Fairseq flag |
| --- | --- |
| `learning_rate` | `--lr` |
| `optimizer` | `--optimizer` |
| `criterion` | `--criterion` |
| `max_epochs` | `--max-epoch` |
| `batch_size` | `--batch-size` |
| `max_tokens` | `--max-tokens` |
| `gradient_clip_val` | `--clip-norm` |
| `accumulate_grad_batches` | `--update-freq` |
| `patience` | `--patience` |
| `seed` | `--seed` |
| `monitor` | `--best-checkpoint-metric` |
| `num_workers` | `--num-workers` |

Anything Fairseq supports that isn't in the table you pass directly through `fairseq_args`.

!!! danger "`fairseq_args` always win on collision"
    If you set `max_epochs=10` *and* `--max-epoch 15` in `fairseq_args`, the run uses **15**.
    This is intentional — it lets you express anything Fairseq supports without AutoNMT needing
    to model it — but it means you must avoid setting the same thing two ways by accident. A
    handful of flags AutoNMT manages itself (`--save-dir`, `--tensorboard-logdir`, `--bpe`,
    `--remove-bpe`, …) are **reserved** and rejected if you pass them.

## When you'd still use it

Only to reproduce or extend an existing Fairseq-based result. For anything new, the
[AutoNMT backend](../toolkit/overview.md) is the maintained path.
