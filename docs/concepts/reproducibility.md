# Reproducibility

A research result you can't reproduce is a result you can't trust. AutoNMT is built so an
experiment is auditable after the fact, without you having to remember to instrument it.

## Persisted artifacts

Every intermediate is written to a numbered stage folder (see
[On-disk layout](on-disk-layout.md)). You can inspect, reuse, or pin any step — the data
that produced a number still exists on disk after the run.

## Effective config dumps

Every run dumps its **full effective config** to its logs:

```text
models/<toolkit>/runs/<run>/logs/
├── config_train.json     # everything fit() actually used
└── config_predict.json   # everything predict() actually used
```

This is the resolved config *after* merging `FitConfig`/`PredictConfig` with any override
kwargs — so it records what the run really did, not what you intended.

!!! note "Only primitives round-trip"
    The config dump filters out non-primitive values. Don't expect arbitrary Python objects
    (a custom callable, a model instance) to appear in the JSON — log those yourself if you
    need them.

## One seed for everything

`manual_seed` seeds Python's `random`, NumPy, Torch, and PyTorch Lightning together, so you
don't have to chase down four separate seeding APIs:

```python
from autonmt.utils.seed import manual_seed
manual_seed(42)
```

In practice you pass `seed=` through the fit config and AutoNMT seeds before training:

```python
trainer.fit(train_ds, config=FitConfig(max_epochs=10, seed=42))
```

## Standard reference tools

AutoNMT delegates the parts that *must* be comparable across papers to widely-used
reference implementations rather than reinventing them:

- **SentencePiece** — subword tokenization
- **sacreBLEU** — BLEU / chrF with versioned, reproducible signatures
- **Moses** (`sacremoses`) — word pretokenization
- **COMET**, **BERTScore** — neural metrics

Because these are the same tools the rest of the field uses, your numbers line up with
published baselines.

## Statistical significance

Beyond point estimates, AutoNMT ships a significance-testing helper in
[`autonmt.evaluation.significance`](../reference/evaluation.md) so you can check whether the
gap between two systems is real or noise before you report it.
