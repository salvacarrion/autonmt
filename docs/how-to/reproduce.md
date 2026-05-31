# Reproduce an experiment

AutoNMT makes a run [reproducible by construction](../concepts/reproducibility.md): staged
artifacts, a dumped effective config, an environment/git snapshot, and a single seed. This
recipe covers the parts you drive: seeding one run, running multiple seeds for variance, and
testing whether a difference is real.

## Seed a single run

```python
trainer.fit(train_ds, config=FitConfig(seed=42))
```

`seed=` calls `manual_seed`, which seeds Python, NumPy, PyTorch, and Lightning together. For
determinism *before* `fit` (vocab building, custom prep), call it yourself first:

```python
from autonmt.utils.seed import manual_seed
manual_seed(seed=42)
```

The run's `logs/config_train.json` records the exact effective config plus the environment
(package versions, git SHA + dirty flag), so the settings are recoverable months later — see
[Configuration & reproducibility](../guide/experiments/configuration.md).

## Run multiple seeds for variance

A single seed removes *avoidable* variance, but GPU non-determinism remains — two runs of the
same config routinely differ by 0.5–1.5 BLEU. For a claim, run a few seeds and report
**mean ± std**. AutoNMT keeps this as a plain loop, each seed in its own `run_name`:

```python
import statistics
from autonmt.backends import AutonmtTranslator
from autonmt.core.nn.models import Transformer

runs_dir = train_ds.get_runs_path(toolkit="autonmt")
seed_bleu = []
for seed in (42, 43, 44):
    model = Transformer.from_vocabs(src_vocab, tgt_vocab)          # fresh weights per seed
    t = AutonmtTranslator(
        model=model, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        runs_dir=runs_dir, run_name=train_ds.get_run_name(run_prefix=f"exp_s{seed}"),
    )
    t.fit(train_ds, max_epochs=10, seed=seed, save_best=True)
    s = t.predict(test_ds, beams=[5], metrics={"bleu"}, eval_mode="compatible", load_checkpoint="best")
    seed_bleu.append(s[0]["translations"]["beam5"]["sacrebleu_bleu_score"])

print(f"BLEU: mean={statistics.mean(seed_bleu):.2f}, std={statistics.stdev(seed_bleu):.2f}")
```

## Test whether a difference is real

For comparing two systems on the *same* test set, follow up with a paired bootstrap on the
`hyp.txt` / `ref.txt` files the runs produced:

```python
from autonmt.evaluation.significance import paired_bootstrap_bleu
result = paired_bootstrap_bleu(hyp_baseline, hyp_yours, ref, n_samples=1000)
print(result["delta"], result["p_value"])
```

The full reasoning is in [Statistical significance](../guide/evaluation/significance.md).

!!! tip "Reproducing someone else's run"
    Open their `config_train.json` / `config_predict.json`: it lists every effective setting
    and the git SHA the run used. Check out that commit, recreate the config, and re-run —
    the staged data and seed make the rest deterministic up to hardware.
