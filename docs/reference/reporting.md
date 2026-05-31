# Reporting

The `Report` / `DatasetReport` classes, the per-run report schema, and the
low-level plot primitives. See [Reports & plots](../guide/evaluation/reports.md)
for usage.

## Report

`Report` (experiment results) and `DatasetReport` (corpus diagnostics), plus the
thin score transforms they wrap.

::: autonmt.reporting.report

## Schema

The per-run report dict each backend emits (`RunMetadata`, `build_run_report`),
upstream of any `Report`.

::: autonmt.reporting.schema

## Plots

Low-level plotting primitives: the `BasePlot` template hierarchy and `PlotStyle`.

::: autonmt.reporting.plots
