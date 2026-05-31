# Configuration & reproducibility

`fit` and `predict` are configured the same way, and every run records exactly how it was
configured. This page covers the config objects, how their values are merged, what gets
persisted, and how seeding and stage-skipping make a re-run cheap and faithful. The full
on-disk tree and the design rationale live in
[Concepts → On-disk layout](../../concepts/on-disk-layout.md) and
[Reproducibility model](../../concepts/reproducibility.md); here we stay practical.

## The config objects

Two typed configs carry everything tunable:

- [`FitConfig`](../training/training.md) — training: epochs, batch size, optimizer, LR
  schedule, criterion, callbacks, loggers, seed, bucketing.
- [`PredictConfig`](../translation/generating.md) — inference: beam widths, metrics,
  checkpoint to load, `eval_mode`, the predict-time preprocessing hook.

```python
from autonmt.backends._base.config import FitConfig, PredictConfig

trainer.fit(train_ds, config=FitConfig(max_epochs=10, batch_size=128, learning_rate=1e-3, seed=42))
scores = trainer.predict(test_ds, config=PredictConfig(beams=[1, 5], metrics={"bleu", "chrf"}))
```

The full field lists belong to the [Training](../training/training.md) and
[Translation](../translation/generating.md) pages; this page is about how those values are
applied and remembered.

## Precedence: defaults < `config=` < kwargs

Both verbs accept a typed config **and** loose keyword arguments, and you can mix them. When
the same key is set in more than one place, the more specific source wins:

```python
# max_epochs=20 (kwarg) overrides the 10 in the config; batch_size=128 stands.
trainer.fit(train_ds, config=FitConfig(max_epochs=10, batch_size=128), max_epochs=20)
```

The order is **library defaults < `config=` object < explicit kwargs**. This lets you keep
a base config for a sweep and override one knob per run without copying the whole object.

## Every run records itself

Reproducibility in AutoNMT is a side effect of how runs are written, not a checklist. On
every `fit()` / `predict()`, the translator dumps the **fully-merged, effective config** to
the run's `logs/`:

```
models/autonmt/runs/<run_name>/logs/
├── config_train.json     # what fit() actually ran with
└── config_predict.json   # what predict() actually ran with
```

Each dump also embeds an **environment snapshot** — Python version; the versions of `torch`,
`pytorch_lightning`, `transformers`, `sentencepiece`, `sacrebleu`, `autonmt`; the working
directory and `argv`; and, when the cwd is a git repo, the **commit SHA and a dirty flag**:

```json
{
  "environment": {
    "python": "3.12.4", "torch": "2.4.0", "autonmt": "1.0.0",
    "git_sha": "26436ac…", "git_dirty": false
  },
  "fit": { "max_epochs": 10, "batch_size": 128, "learning_rate": 0.001 }
}
```

!!! note "Configs keep primitives only"
    The dump renders non-primitive values readably — a callable hook becomes its
    `module.qualname`, so you always see *which* `preprocess_fn` or decoder a run used — but
    it doesn't try to serialize arbitrary objects. The object itself won't round-trip.

## One seed for everything

```python
trainer.fit(train_ds, config=FitConfig(seed=42))
```

Passing `seed=` makes `fit` call `manual_seed(42)`, which seeds Python's `random`, NumPy,
PyTorch, and Lightning together. For determinism *before* `fit` (vocab construction, custom
data prep), call it yourself up front:

```python
from autonmt.utils.seed import manual_seed
manual_seed(seed=42)
```

!!! tip "Seeds don't make neural training bit-identical"
    A single seed removes the *avoidable* variance (data order, init), but GPU
    non-determinism and hardware differences remain. For publication-grade comparisons, run
    multiple seeds and report mean ± std — and use a [significance
    test](../evaluation/significance.md) for close results. AutoNMT keeps multi-seed as a
    `for` loop in *your* code (each seed gets its own `run_name`); see
    [How-to → Reproduce an experiment](../../how-to/reproduce.md).

## Stage skipping: why re-runs are cheap

Every data stage and the training step check whether their output already exists before
doing any work, gated by `force_overwrite`:

```python
builder.build(force_overwrite=False)   # default: keep existing artifacts, skip done stages
builder.build(force_overwrite=True)    # rebuild everything from scratch
```

Run the same script twice and the second run skips preprocessing, encoding, vocab building,
and even training when a checkpoint already exists — it jumps straight to whatever is
missing. To force-rebuild one stage, delete **only that stage's folder** and re-run:

```bash
rm -r data/multi30k/de-en/original/data/4_encoded/bpe/4000   # then re-run
```

!!! warning "Delete the stage, not the tree"
    Because each stage is independent on disk, the right way to rebuild (say) the encoding
    is to remove `4_encoded/<subword>/<vocab>/` — not to nuke the whole dataset. Checkpoints
    get the same care: training renames an existing `.pt` to `.pt.bak` rather than clobbering
    it when `force_overwrite=True`.

---

That's the reproducibility surface you touch day to day. For the complete directory tree
and the reasoning behind it, continue to [Concepts → On-disk layout](../../concepts/on-disk-layout.md).
