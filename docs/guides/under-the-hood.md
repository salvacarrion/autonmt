# Under the hood

The convenience one-liners - `AutonmtTranslator.from_dataset(...)`,
`ds.build_vocabs(...)`, `Transformer.from_vocabs(...)`, `trainer.predict(...)`,
`generate_report(...)` - each have a lower-level equivalent. This guide shows when to reach
for them. Mirrors
[`examples/advanced/01_under_the_hood.py`](https://github.com/salvacarrion/autonmt/blob/main/examples/advanced/01_under_the_hood.py).

You don't need any of this for normal use. Reach for it when you want to:

- sanity-check splits for **train/test leakage** before training,
- swap a single piece (custom decoder, custom callbacks, custom Transformer dims),
- **resume from a checkpoint** instead of retraining,
- run translate / score / parse independently (e.g. re-score with a new metric without
  re-translating),
- assemble the report **manually** from score dicts (to add columns or merge experiments),
- run **multi-seed** experiments and aggregate variance for publication.

## Build vocabularies explicitly

`ds.build_vocabs()` wraps the construction of two
[`Vocabulary`][autonmt.vocabularies.whitespace_vocab.Vocabulary] objects from the SPM
artifacts on disk. You can build them directly when you need finer control.

```python
from autonmt.vocabularies import Vocabulary
# load from the cell's trained vocab artifacts, capping size as needed
```

## Check for leakage

Before training on splits you assembled, verify the test set didn't leak into train:

```python
from autonmt.datasets.leakage import warn_on_leakage, find_leaked_lines

warn_on_leakage(train_ds)             # logs a warning
leaked = find_leaked_lines(train_ds)  # returns the offending lines for inspection
```

## Decode, score, and report separately

`predict()` bundles three steps - translate, score, build the report dict. Splitting them
lets you, for instance, re-score existing hypotheses with a new metric without paying for
decoding again. The report builders are public:

```python
from autonmt.reporting.report import scores_to_dataframe, summarize_scores, format_summary_table

df_report  = scores_to_dataframe(scores)   # flatten the list of score dicts
df_summary = summarize_scores(df_report)    # keep ids + reference-metric columns
print(format_summary_table(df_summary))
```

This is exactly what [`generate_report`][autonmt.reporting.report.generate_report] does
internally - calling them yourself lets you add custom columns or merge with another
experiment's dataframe before writing anything to disk.

## Custom decoding

Swap the decoding strategy by importing one of the search algorithms and wiring it into the
run:

```python
from autonmt.core.decoding import BeamSearch
# configure beam size, length penalty, etc., then pass it through predict
```

See [`autonmt.core.decoding`](../reference/core.md) for `GreedySearch`, `BeamSearch`,
`MultinomialSampling`, `TopkSampling`, `ToppSampling`.

## Multi-seed variance

For publication-grade comparisons, run each cell under several seeds and aggregate. Seed
everything with one call and loop:

```python
import statistics
from autonmt.utils.seed import manual_seed

bleus = []
for seed in (42, 43, 44):
    manual_seed(seed)
    trainer.fit(train_ds, config=FitConfig(max_epochs=10, seed=seed))
    s = trainer.predict(test_datasets, config=pred_cfg)
    bleus.append(s[...]["translations"]["beam5"]["sacrebleu_bleu_score"])

print(f"BLEU {statistics.mean(bleus):.2f} ± {statistics.stdev(bleus):.2f}")
```

Pair this with the paired-bootstrap test in
[`autonmt.evaluation.significance`](../reference/evaluation.md) to check whether a gap
between two systems is statistically real.

## Resume from a checkpoint

`PredictConfig(load_checkpoint="best")` already loads the best checkpoint a run saved. To
evaluate a previously-trained run without retraining, construct the translator and call
`predict()` directly - the checkpoints persist on disk under the run folder (see
[On-disk layout](../concepts/on-disk-layout.md)).

## Other advanced examples

The [`examples/advanced/`](https://github.com/salvacarrion/autonmt/tree/main/examples/advanced)
folder goes further: LoRA fine-tuning, catastrophic-forgetting studies, and model merging.
