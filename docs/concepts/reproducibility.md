# Reproducibility model

Reproducibility in AutoNMT isn't a checklist you remember to follow — it's a property of how
runs are written. The [on-disk layout](on-disk-layout.md) gives you staged, path-driven
artifacts; this page covers the other half: how each run records *what produced it*, and how
seeding works. The day-to-day knobs are on [Configuration &
reproducibility](../guide/experiments/configuration.md); this is the model behind them.

## Every run records itself

On every `fit()` / `predict()`, `BaseTranslator`:

- **Dumps the effective config** to `logs/config_train.json` / `config_predict.json` — the
  fully-merged settings (defaults < `config=` < kwargs), so there's no ambiguity about which
  values actually ran.
- **Snapshots the environment** into that config: Python version, the versions of `torch`,
  `pytorch_lightning`, `transformers`, `sentencepiece`, `sacrebleu`, `autonmt`, plus the
  working directory, `argv`, and — when the cwd is a git repo — the **commit SHA and a dirty
  flag**.

```json
{
  "environment": {
    "python": "3.12.4", "torch": "2.4.0", "autonmt": "1.0.0",
    "git_sha": "26436ac…", "git_dirty": false
  },
  "fit": { "max_epochs": 5, "batch_size": 128, "learning_rate": 0.001 }
}
```

!!! note "Configs keep primitives only"
    The config dump renders non-primitive values readably (a callable hook becomes
    `module.qualname`) but doesn't try to serialize arbitrary objects. So you'll always see
    *which* `preprocess_fn` or decoder a run used, but the object itself won't round-trip.

## One seed for everything

```python
trainer.fit(train_ds, config=FitConfig(seed=42))
```

Passing `seed=` makes `fit` call `manual_seed(42)`, which seeds Python's `random`, NumPy,
PyTorch, and Lightning together. For full determinism *before* `fit` (vocab construction,
custom data prep), call it yourself up front:

```python
from autonmt.utils.seed import manual_seed
manual_seed(seed=42)
```

!!! tip "Seeds don't make neural training bit-identical"
    A single seed removes the *avoidable* variance (data order, init), but GPU
    non-determinism and hardware differences remain. For publication-grade comparisons, run
    multiple seeds and report mean ± std — and use a [significance
    test](../guide/evaluation/significance.md) for close results. AutoNMT deliberately leaves
    multi-seed as a `for` loop in *your* code (each seed gets its own `run_name`); see
    [How-to → Reproduce an experiment](../how-to/reproduce.md).

## Why this composes with everything else

Because a run is fully described by its staged inputs + its dumped config + its seed, the
three layers reinforce each other: the layout says *where*, the config dump says *with what*,
and the seed pins *the randomness*. Hand someone the `base_path` and the commit SHA and they
can reconstruct the run up to hardware non-determinism — without you writing a single
bespoke logging line.

---

That completes the conceptual tour. To put it to work, the [User
guide](../guide/experiments/workflow.md); for exact signatures, the
[API reference](../reference/index.md).
