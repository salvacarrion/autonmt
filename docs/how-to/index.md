# How-to guides

The [User guide](../guide/experiments/workflow.md) explains each part of AutoNMT
systematically. These **how-to guides** are the opposite cut: each one solves a single,
concrete task that usually spans several parts of the pipeline. They assume you've met the
concepts already and want the recipe.

## Running experiments

- [**Run a complete experiment**](complete-experiment.md) — build → train → translate →
  report, as one script you can adapt.
- [**Compare multiple models**](compare-models.md) — several architectures, one dataset, one
  comparable table.
- [**Use a custom dataset**](custom-dataset.md) — bring your own parallel files (or a Hub
  corpus) into the AutoNMT layout.
- [**Run with another backend**](swap-backend.md) — the same experiment on the native engine,
  HuggingFace, or Fairseq — and mixing them in one report.
- [**Reproduce an experiment**](reproduce.md) — seeds, multi-seed variance, and significance
  testing for publication-grade results.

## Extending the framework

- [**Add a custom model**](custom-model.md) — a new architecture as a small subclass.
- [**Change the decoding strategy**](custom-decoding.md) — a new search/sampling rule.
- [**Add a custom metric**](custom-metric.md) — register a metric backend.

## Going low-level

- [**Drive the pipeline manually**](manual-pipeline.md) — unpack the convenience layer:
  manual vocab/model/translator construction, split translate→score→parse, build the report
  by hand.

!!! tip "Extension is the design, not an escape hatch"
    Custom models, decoders, and metrics aren't workarounds — AutoNMT's
    [minimal core](../concepts/philosophy.md#extensible) is *built* to be extended by
    subclassing and registration, so these recipes are first-class usage.
