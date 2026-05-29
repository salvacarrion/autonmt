# Fairseq backend

!!! warning "Deprecated"
    Fairseq was [archived by its maintainers on 2026-03-20](https://github.com/facebookresearch/fairseq)
    and no longer receives updates.
    [`FairseqTranslator`][autonmt.backends.fairseq.translation_engine.FairseqTranslator] is
    kept working for users with existing flows, but **new projects should use the
    [AutoNMT backend](autonmt.md)** (PyTorch Lightning).

    Importing the module emits a `DeprecationWarning`; instantiating it without `fairseq`
    installed raises `ImportError` with install instructions. Fairseq is **not** in the
    default dependencies — install it with `pip install -e '.[fairseq]'`.

`FairseqTranslator` shells out to the Fairseq CLI. AutoNMT translates its kwargs
(`max_epochs`, `batch_size`, …) into Fairseq flags via an internal `_AUTONMT_TO_FAIRSEQ`
table.

## Usage

```python
from autonmt.backends.fairseq.translation_engine import FairseqTranslator  # DeprecationWarning
from autonmt.backends._base.config import FitConfig

# Vocabs are still needed: the base translator encodes the eval splits with the
# same subword model the training run used.
src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)

trainer = FairseqTranslator.from_dataset(
    train_ds, src_vocab=src_vocab, tgt_vocab=tgt_vocab, run_prefix="fairseq",
)
trainer.fit(
    train_ds,
    config=FitConfig(max_epochs=5, batch_size=128),
    fairseq_args=["--arch transformer", "--dropout 0.1"],
)
```

## Precedence: `fairseq_args` always win

!!! danger "On collision, raw Fairseq flags override AutoNMT kwargs"
    If you set `max_epochs=10` *and* `--max-epoch 15` in `fairseq_args`, the run uses **15**.

    This is intentional: it lets you express anything Fairseq supports without AutoNMT
    needing to know about it. The trade-off is that you must avoid setting the same thing
    two ways by accident.

## When you'd still use it

Only to reproduce or extend an existing Fairseq-based result. For anything new, the AutoNMT
Lightning backend gives you the same `fit()` / `predict()` surface with a maintained engine,
in-process models, and custom-model support.

## API reference

See [`FairseqTranslator`][autonmt.backends.fairseq.translation_engine.FairseqTranslator] in
the [backends API reference](../reference/backends.md#fairseqtranslator).
