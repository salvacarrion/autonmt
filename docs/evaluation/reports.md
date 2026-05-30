# Reports & plots

The whole point of running a [grid](../introduction/philosophy.md#grid-first) is to compare
its cells *side by side*. The reporting layer turns the score dicts
[`predict`](../toolkit/predict.md) returns into JSON + CSV artifacts and comparison plots —
every run scored identically, in one table.

## `generate_report`

The one call that does it all:

```python
from autonmt.reporting.report import generate_report

df_report, df_summary = generate_report(
    scores=all_scores,                 # list of predict() results (see shape below)
    output_path="outputs/sweep",
    plot_metric="translations.beam5.sacrebleu_bleu_score",   # optional comparison plot
)
```

It writes, under `output_path/`:

```text
reports/report.json          # the full nested score dicts
reports/report.csv           # flattened: one row per (run, eval_ds)
reports/report_summary.csv   # identifying columns + the reference metric
plots/plot__<metric>.png     # grouped bar chart (only if plot_metric is given)
```

…and returns two DataFrames: `df_report` (everything) and `df_summary` (the trimmed table).

### The `scores` shape

`predict` returns a **list** (one entry per evaluation set). `generate_report` expects a
**list of those** — one per run — so you nest:

```python
all_scores = []
for train_ds in builder.get_train_ds():
    trainer = AutonmtTranslator.from_dataset(train_ds, ...)
    trainer.fit(train_ds, ...)
    all_scores.append(trainer.predict(builder.get_test_ds(), ...))   # one run

generate_report(scores=all_scores, output_path="outputs/sweep")
```

For a single run, wrap it once: `generate_report(scores=[scores], ...)`.

## The score key schema

Every metric value is flattened into a key of the form
**`<tool>_<metric>_<field>`**, nested under the beam it came from:

```text
translations.beam5.sacrebleu_bleu_score     # BLEU at beam width 5
translations.beam5.sacrebleu_chrf_score      # chrF at beam width 5
translations.beam1.comet_comet_score         # COMET, greedy
translations.beam5.bertscore_bertscore_f1    # BERTScore F1
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
table.

## Reading the summary in the terminal

```python
from autonmt.reporting.report import format_summary_table

print(format_summary_table(df_summary))
```

`format_summary_table` renders the summary DataFrame as a tidy, width-aware table with
thousands-separated ints and two-decimal floats — handy for a quick look without opening the
CSV.

## Building reports by hand

`generate_report` is a convenience over three small, public transforms you can call yourself
to merge experiments, add columns, or skip disk artifacts:

```python
from autonmt.reporting.report import scores_to_dataframe, summarize_scores

df_report = scores_to_dataframe(all_scores)      # flatten the nested dicts → DataFrame
df_summary = summarize_scores(df_report)         # keep id columns + the reference metric
```

The full manual workflow (transform → save → plot, step by step) is in [Full manual
control](../toolkit/manual-control.md#build-the-report-by-hand).

## Plots

### Model comparison

`generate_report(..., plot_metric=...)` draws a grouped bar chart comparing one metric across
runs. You can also call it directly for more control:

```python
from autonmt.reporting.figures import plot_model_comparison

plot_model_comparison(
    df_report, out_dir="outputs/sweep/plots",
    metric="translations.beam5.sacrebleu_bleu_score",
    xlabel="Model", ylabel="BLEU", title="de→en comparison",
)
```

### Metric sweeps

When you've swept a single axis (vocab size, model size), `generate_sweep_report` /
`plot_metric_sweep` draws a line plot of a metric against that axis — ideal for "BLEU vs vocab
size" curves, with an optional second y-axis:

```python
from autonmt.reporting.report import generate_sweep_report

generate_sweep_report(
    data=df, output_path="outputs/sweep", x="vocab__size",
    y_left="translations.beam5.sacrebleu_bleu_score",
    save_csv=True,
)
```

### Dataset diagnostics { #dataset-diagnostics }

Independent of any model, you can plot the **data itself** — sentence-length distributions,
split sizes, and the vocabulary frequency distribution — for any prepared dataset variant:

```python
from autonmt.reporting.figures import plot_dataset_diagnostics

for ds in builder.get_train_ds():
    plot_dataset_diagnostics(ds, merge_vocabs=False)   # writes into ds.get_plots_path()
```

This is intentionally **not** part of `DatasetBuilder.build()` (plotting isn't a build
responsibility) — call it after building when you want the figures. They're useful for
sanity-checking preprocessing before you spend GPU time.

---

That completes the pipeline end to end. To extend any layer with your own component, see
**[Extending AutoNMT](../extending/index.md)**.
