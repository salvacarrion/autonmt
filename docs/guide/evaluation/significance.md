# Statistical significance

A **+0.4 BLEU** improvement means little without knowing whether it's within run-to-run
noise. Reviewers in ACL/EMNLP/WMT increasingly require a significance test, and AutoNMT
provides one directly on the `hyp.txt` / `ref.txt` files [`predict`](../translation/generating.md)
already produced.

!!! info "Why a significance test at all?"
    Two systems evaluated on the *same* finite test set will differ by some amount even if
    they're equally good — the test set is a sample, not the whole language. A significance
    test asks: *if I'd drawn a slightly different test set, would system B still beat system
    A?* Without it, small BLEU gaps are indistinguishable from luck.

## Paired bootstrap resampling

AutoNMT's test is the paired bootstrap of [Koehn (2004)](https://aclanthology.org/W04-3250/),
the standard significance test for machine translation.

```python
from autonmt.evaluation.significance import paired_bootstrap_bleu
from autonmt.utils import fileio

hyp_a = fileio.read_file_lines("…/baseline/…/beam5/hyp.txt", autoclean=True)
hyp_b = fileio.read_file_lines("…/yours/…/beam5/hyp.txt",    autoclean=True)
ref   = fileio.read_file_lines("…/yours/…/beam5/ref.txt",    autoclean=True)

result = paired_bootstrap_bleu(hyp_a, hyp_b, ref, n_samples=1000)
print(result)  # {bleu_a, bleu_b, delta, p_value, n_samples}
```

!!! info "What a paired bootstrap tells you"
    It resamples the test set with replacement many times and re-scores **both** systems on
    each resample. Because they see the *same* resamples, the variance isolates the systems'
    difference from the noise of the test set itself. The returned `p_value` is the fraction
    of resamples where system B did **not** beat A (one-sided H₀: `score(B) ≤ score(A)`). A
    small `p_value` (e.g. < 0.05) means the improvement is unlikely to be noise. Use
    `n_samples ≥ 1000` for reportable results.

## Pair it with multiple seeds

A significance test on one pair of runs controls for *test-set* noise; it does not control
for *training* noise (initialization, data order). For publication-grade claims, also run
each system across several seeds and report **mean ± std**, then use the bootstrap for the
close comparisons. AutoNMT keeps multi-seed as a plain `for` loop in your code — each seed
gets its own `run_name` — so it composes naturally with the report:

```python
for seed in (1, 2, 3):
    trainer = AutonmtTranslator.from_dataset(train_ds, ..., run_prefix=f"sys-b-seed{seed}")
    trainer.fit(train_ds, config=FitConfig(seed=seed))
    ...
```

The full pattern is in [How-to → Reproduce an experiment](../../how-to/reproduce.md).

---

Next: collecting all of these scores into one comparable report —
**[Reports & plots](reports.md)**.
