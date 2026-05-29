# Evaluation

Metric backends and statistical significance testing.

## Metrics

Each backend in the registry knows how to both compute and parse its own score artifact, so
there's no parallel parser table to keep in sync. Metric strings prefixed `hg_` are
dispatched to HuggingFace `evaluate`.

::: autonmt.evaluation.metrics

## Significance testing

::: autonmt.evaluation.significance
