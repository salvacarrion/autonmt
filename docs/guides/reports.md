# Reports & plots

The third pipeline layer turns the score dicts your `predict()` calls return into a single
comparable table plus figures.

## generate_report

[`generate_report`][autonmt.reporting.report.generate_report] takes the **list** of
per-run score dicts and writes artifacts under `<output_path>/reports/`:

```python
from autonmt.reporting.report import generate_report, format_summary_table

df_report, df_summary = generate_report(
    scores=scores,                       # list of dicts from predict()
    output_path="outputs/run1",
    plot_metric="translations.beam5.sacrebleu_bleu_score",  # optional → also plots
)
print(format_summary_table(df_summary))
```

It returns `(df_report, df_summary)` and writes three files:

| File                         | Contents                                                        |
| ---------------------------- | --------------------------------------------------------------- |
| `reports/report.json`        | The full, unflattened per-run dicts                             |
| `reports/report.csv`         | Every flattened score column, one row per (run, test set)       |
| `reports/report_summary.csv` | Identifying columns + the columns matching the reference metric |

If you pass `plot_metric`, a model-comparison figure is also written under `plots/`.

## The score schema

Score keys are flattened as `translations.beam<n>.<tool>_<metric>_<field>`. Examples:

- `translations.beam1.sacrebleu_bleu_score`
- `translations.beam5.sacrebleu_chrf_score`
- `translations.beam5.bertscore_f1_mean`

Because every backend emits the same schema, an AutoNMT model, a fine-tuned HuggingFace
checkpoint, and a Fairseq run all land in the **same table** and are directly comparable.

A typical `df_summary` (truncated):

```text
train_dataset  test_dataset  vocab__subword_model  vocab__size  model__architecture  translations.beam5.sacrebleu_bleu_score
multi30k       multi30k      unigram               4000         transformer                                        35.123375
multi30k       multi30k      word                  4000         transformer                                        34.706139
```

## Plots

The figure helpers live in [`autonmt.reporting.figures`](../reference/reporting.md).

### Model comparison

Bar/grouped comparison of cells for one metric:

```python
from autonmt.reporting.figures import plot_model_comparison

plot_model_comparison(
    df_report=df_report, out_dir="outputs/run1/plots",
    metric="translations.beam5.sacrebleu_bleu_score",
    xlabel="Train variant", ylabel="BLEU", title="Vocab-size sweep",
)
```

### Metric sweeps

For "X vs Y" line plots (e.g. average tokens vs BLEU), use
[`generate_sweep_report`][autonmt.reporting.report.generate_sweep_report] /
`plot_metric_sweep`, which support a left and optional right Y-axis.

### Dataset diagnostics

`plot_dataset_diagnostics` produces sentence-length histograms, split-size bars, and
vocabulary-distribution plots straight from a `Dataset` - useful for sanity-checking a
corpus before you train on it. The low-level primitives live in
[`autonmt.reporting.plots`](../reference/reporting.md).

## Styling

Every plotting function accepts a `style` argument (a `PlotStyle`); pass your own to match
a paper's figure conventions, or rely on `DEFAULT_STYLE`. Extra keyword arguments to
`generate_report` are forwarded to `plot_model_comparison` (e.g. `group_label_fn`,
`legend_label_fn`).
