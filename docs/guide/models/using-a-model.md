# Using a model

A model is the *one part of an experiment your paper is actually about* — everything else
in AutoNMT is scaffolding around it. The native engine ships a handful of ready-to-train
encoder–decoder architectures; this page is about picking one, sizing it to your data, and
handing it to a translator. The full inventory is in the [Model catalog](catalog.md), the
reusable layers in [Building blocks](building-blocks.md), and writing your own in
[Writing your own model](custom-models.md).

## Instantiate from vocabularies

Every built-in lives under `autonmt.core.nn.models`. The convenience constructor
`from_vocabs` infers the source/target sizes and the pad id from the vocabularies, so you
don't repeat them:

```python
from autonmt.core.nn.models import Transformer

src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)

model = Transformer.from_vocabs(src_vocab, tgt_vocab)                       # all defaults
model = Transformer.from_vocabs(src_vocab, tgt_vocab,                       # override anything
                                encoder_layers=6, decoder_layers=6, dropout=0.3)
```

`from_vocabs` asserts `src_vocab.pad_id == tgt_vocab.pad_id` (true by default); if you build
vocabularies with different pad ids, pass `padding_idx=` to the plain constructor instead.

!!! info "Sizing knobs are shared across architectures"
    The Transformer and RNN families share the same vocabulary-driven sizing pattern and a
    common set of dimensions (`encoder_embed_dim`, `*_layers`, `dropout`, …). Defaults build
    a *small* model that trains in minutes — fine for smoke tests, deliberately under-sized
    for real results. Scale the dimensions up (and the data with them) for publishable runs.

## Plug it into a translator

The model is inert until a [translator](../backends/native.md) owns it. With the native
backend:

```python
from autonmt.backends import AutonmtTranslator

trainer = AutonmtTranslator.from_dataset(
    train_ds,
    model=model,
    src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    run_prefix="exp",
)
trainer.fit(train_ds, config=FitConfig(max_epochs=10))
```

The same model object can be reused across runs, or rebuilt per cell inside the
[experiment loop](../experiments/workflow.md). Swapping to the HuggingFace backend means
*not* building a native model at all — you pass a `model_id` instead (see
[HuggingFace](../backends/huggingface.md)).

## What you get for free

Because every architecture subclasses the shared seq2seq base, you inherit the entire
training/validation machinery without writing any of it:

- **Optimizer & LR schedule** — resolved from [`FitConfig`](../training/training.md)
  (`optimizer="adamw"`, `scheduler="noam"`, …), with per-step scheduler updates.
- **Teacher-forced training & validation steps** — cross-entropy that ignores `<pad>`, plus
  optional decoded-sample logging.
- **`count_parameters()`** — total / trainable / non-trainable counts, also recorded in the
  run metadata.
- **KV-cached incremental decoding** — the Transformer and RNNs advertise
  `supports_incremental_decoding`, so beam search reuses cached state instead of recomputing
  (details in [Building blocks](building-blocks.md)).

---

Next: the full **[Model catalog](catalog.md)** — what each architecture is and when to reach
for it.
