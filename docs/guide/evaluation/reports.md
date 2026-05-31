# Reports & plots

The whole point of running a [grid](../../concepts/philosophy.md#grid-first) is to compare
its cells *side by side*. The reporting layer turns the score dicts
[`predict`](../translation/generating.md) returns into JSON + CSV artifacts and comparison
plots — every run scored identically, in one table.

## `Report`

`Report` wraps the scores and presents a small, chainable API:

```python
from autonmt.reporting.report import Report

report = (
    Report.from_runs(all_scores, output_path="outputs/sweep")  # see the shape below
    .save()                                                    # JSON + CSV artifacts
    .plot_comparison("bleu", beam=5)                           # grouped bar chart
)
print(report)                                                  # tidy terminal table
```

`save()` writes, under `output_path/`:

```text
reports/report.json          # the full nested score dicts
reports/report.csv           # flattened: one row per (run, eval_ds)
reports/report_summary.csv   # identifying columns + the reference metric
```

and each `plot_*` writes a figure under `plots/`. Every `save` / `plot_*` returns the
`Report`, so calls chain. The wide and trimmed DataFrames are always available as
`report.df` and `report.summary`.

### The `scores` shape

`predict` returns a **list** (one entry per evaluation set) — that's one *run*. Use the
constructor that matches what you have:

```python
# One run (one trained model):
report = Report.from_predict(scores, output_path="outputs/run")

# A grid (many runs) — nest one predict() result per run:
all_scores = []
for train_ds in builder.get_train_ds():
    trainer = AutonmtTranslator.from_dataset(train_ds, ...)
    trainer.fit(train_ds, ...)
    all_scores.append(trainer.predict(builder.get_test_ds(), ...))   # one run
report = Report.from_runs(all_scores, output_path="outputs/sweep")

# Or accumulate incrementally:
report = Report.from_predict(scores, output_path="outputs/sweep")
report.add(more_scores)
```

## Naming metrics

You **don't** write the long column name. Ask for a metric by a short, generic name and
`Report` resolves it against the columns that actually exist:

```python
report.plot_comparison("bleu", beam=5)            # → translations.beam5.sacrebleu_bleu_score
report.plot_comparison("chrf", beam=1)
report.plot_comparison("hg_bleu")                 # HuggingFace metrics work too
report.available_metrics()                        # discover what's plottable
```

Resolution is purely pattern-based (no hardcoded metric table), so `hg_*`, `comet`,
`bertscore_f1`, … all work. Disambiguate with:

- `beam=` — required only when several beam widths are present (inferred otherwise).
- `tool=` — picks the tool when two compute the same metric (`sacrebleu` vs `hg`).
- `split=` — picks the [`test_subsets`](../experiments/workflow.md) slice when present.

Ambiguous or unknown names raise a `ValueError` that lists the candidates / available
metrics, so you can narrow the query.

## The score key schema

Every metric value is flattened into a key of the form **`<tool>_<metric>_<field>`**, nested
under the beam (and, with `test_subsets`, the split) it came from:

```text
translations.beam5.sacrebleu_bleu_score        # BLEU at beam width 5
translations.beam5.sacrebleu_chrf_score         # chrF at beam width 5
translations.beam1.comet_comet_score            # COMET, greedy
translations.taskA.beam5.sacrebleu_bleu_score   # BLEU on the "taskA" test subset
```

Alongside the metrics, each row carries identifying columns the report uses to label and
group runs:

| Column | Meaning |
| --- | --- |
| `engine` | `autonmt` / `huggingface` / `fairseq` |
| `run_name` | the run's directory name |
| `train_dataset`, `test_dataset` | dataset names |
| `train__lang_pair`, `test__lang_pair` | language pairs |
| `vocab__subword_model`, `vocab__size` | tokenization of the run |
| `model__architecture`, `model__total_params` | model identity |
| `config` | the full effective config + environment snapshot |

This schema is **identical across backends**, which is why an AutoNMT model, a fine-tuned
HuggingFace checkpoint, and a Fairseq baseline can land in one report and be compared in one
table. (The schema itself lives in `autonmt.reporting.schema`.)

## Reading the summary in the terminal

`print(report)` renders the summary as a tidy, width-aware table with thousands-separated
ints and two-decimal floats — handy for a quick look without opening the CSV. The underlying
`format_summary_table` is also exported if you want to render an arbitrary DataFrame.

## Building reports by hand

`Report` is a thin facade over a few small, public transforms you can call yourself to merge
experiments, add columns, or skip disk artifacts:

```python
from autonmt.reporting.report import scores_to_dataframe, summarize_scores

df_report = scores_to_dataframe(all_scores)      # == report.df
df_summary = summarize_scores(df_report)         # == report.summary
```

The full manual workflow (transform → save → plot, step by step) is in
[How-to → Drive the pipeline manually](../../how-to/manual-pipeline.md).

## Plots

All plot methods return the `Report`, take an optional `out_dir` / `fname`, and resolve the
metric by short name.

### Model comparison

A grouped bar chart comparing one metric across runs:

```python
report.plot_comparison(
    "bleu", beam=5,
    xlabel="Model", ylabel="BLEU", title="de→en comparison",
)
```

### Metric sweeps

When you've swept a single axis (vocab size, model size, a merge weight), draw a line plot of
a metric against that axis — with an optional second y-axis:

```python
report.plot_sweep(
    "bleu", x="vocab__size", hue="test_dataset",
    title="BLEU vs vocab size",
)
```

### Metric matrix

A heatmap of one metric over a `rows` × `cols` grid — e.g. a train × test cross-evaluation:

```python
report.plot_matrix("bleu", rows="train_dataset", cols="test_dataset")
```

### Dataset diagnostics { #dataset-diagnostics }

Independent of any model, plot the **data itself** — sentence-length distributions, split
sizes, and the vocabulary frequency distribution — for any prepared dataset variant:

```python
from autonmt.reporting.report import DatasetReport

for ds in builder.get_train_ds():
    DatasetReport(ds).generate(merge_vocabs=False)   # writes into ds.get_plots_path()
```

This is intentionally **not** part of `DatasetBuilder.build()` (plotting isn't a build
responsibility) — call it after building when you want the figures. They're useful for
sanity-checking preprocessing before you spend GPU time.

---

That completes the pipeline end to end. The same `fit` / `predict` / report flow runs on any
**[backend](../backends/choosing.md)**; to extend a layer with your own component, see the
**[How-to guides](../../how-to/index.md)**.
