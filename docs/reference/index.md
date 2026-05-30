# API reference

This section is generated directly from the source docstrings, so it always matches the
installed version. It's organized by package, following the
[architecture](../concepts/pipeline.md):

<div class="grid cards" markdown>

- :material-database:{ .lg .middle } **[Datasets](datasets.md)**

  ***

  `DatasetBuilder`, `Dataset`, preprocessing, encoding, the HuggingFace loader, and the
  leakage checker.

- :material-format-list-bulleted:{ .lg .middle } **[Vocabularies](vocabularies.md)**

  ***

  `Vocabulary`, the base class, and vocab artifact building.

- :material-swap-horizontal:{ .lg .middle } **[Backends](backends.md)**

  ***

  `BaseTranslator` and the three concrete translators, plus `FitConfig` / `PredictConfig`.

- :material-brain:{ .lg .middle } **[Models & decoding](core.md)**

  ***

  `LitSeq2Seq`, the built-in architectures, decoding strategies, samplers, and the torch
  dataset.

- :material-chart-line:{ .lg .middle } **[Evaluation](evaluation.md)**

  ***

  The `MetricBackend` registry and the significance test.

- :material-file-chart:{ .lg .middle } **[Reporting](reporting.md)**

  ***

  `generate_report`, domain figures, and low-level plot primitives.

</div>

!!! tip "Reading the source"
Every documented symbol has a **source** link. AutoNMT is small enough to read - when in
doubt, follow it.
