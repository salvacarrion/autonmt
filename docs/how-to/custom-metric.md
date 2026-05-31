# Add a custom metric

Metrics are entries in the [`MetricBackend`](../guide/evaluation/metrics.md) registry. A
backend bundles a `compute_fn` (write a score artifact) and a `parse_fn` (read it back into
`{metric: {field: value}}`). Register one and it's available to every translator — no fork,
no parallel parser table.

```python
from autonmt.evaluation.metrics import MetricBackend, METRIC_BACKENDS

def compute_mymetric(*, ref_file, hyp_file, output_file, metrics, **_):
    score = my_scorer(ref_file, hyp_file)
    save_json([{"name": "mymetric", "score": score}], output_file)

def parse_mymetric(text_lines):
    import json
    d = json.loads("".join(text_lines))[0]
    return {"mymetric": {"score": float(d["score"])}}

METRIC_BACKENDS["mymetric"] = MetricBackend(
    name="mymetric", metrics=frozenset({"mymetric"}),
    compute_fn=compute_mymetric, parse_fn=parse_mymetric,
    needs_src=False,    # set True if compute_fn consumes src_file (like COMET)
)
```

Then request it like any built-in metric:

```python
trainer.predict(test, config=PredictConfig(metrics={"bleu", "mymetric"}))
```

The flattened report key follows the standard schema, `mymetric_mymetric_score`, so it lands
in the [report](../guide/evaluation/reports.md) next to BLEU and chrF automatically.

!!! tip "Need a metric that already exists on the HF hub?"
    Don't write a backend — prefix its name with `hg_` (e.g. `hg_meteor`) and AutoNMT loads
    it via `evaluate`. See [HuggingFace metrics](../guide/evaluation/metrics.md#huggingface-metrics).
