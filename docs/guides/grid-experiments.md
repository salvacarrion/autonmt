# Grid experiments

This guide turns the single-cell quickstart into a real sweep. Mirrors
[`examples/basics/04_grid_experiment.py`](https://github.com/salvacarrion/autonmt/blob/main/examples/basics/04_grid_experiment.py)
and [`05_full_grid.py`](https://github.com/salvacarrion/autonmt/blob/main/examples/basics/05_full_grid.py).

Read [The grid](../concepts/grid.md) first for the concepts; this is the hands-on version.

## Sweeping one axis

Start by varying a single thing — say vocabulary size — and keeping everything else fixed:

```python
builder = DatasetBuilder(
    base_path="datasets/sweep",
    datasets=[{"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]}],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [2000, 4000, 8000]}],  # 3 cells
    preprocess_raw_fn=preprocess_train,
    preprocess_splits_fn=preprocess_train,
).build(force_overwrite=False)
```

The experiment loop is axis-agnostic — it iterates whatever the builder produced:

```python
fit_cfg  = FitConfig(max_epochs=2, batch_size=128, learning_rate=1e-3, seed=42)
pred_cfg = PredictConfig(metrics={"bleu", "chrf"}, beams=[5],
                         load_checkpoint="best", preprocess_fn=preprocess_predict,
                         eval_mode="compatible")

scores = []
for train_ds in builder.get_train_ds():
    src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)
    trainer = AutonmtTranslator.from_dataset(
        train_ds, model=Transformer.from_vocabs(src_vocab, tgt_vocab),
        src_vocab=src_vocab, tgt_vocab=tgt_vocab, run_prefix="sweep",
    )
    trainer.fit(train_ds, config=fit_cfg)
    scores.append(trainer.predict(builder.get_test_ds(), config=pred_cfg))
```

## Multiple axes at once

Add entries and the cross-product grows automatically. This declares
`2 sizes × (2 bpe vocabs + 1 char vocab) = 6` cells:

```python
datasets=[{
    "name": "multi30k", "languages": ["de-en"],
    "sizes": [("original", None), ("10k", 10_000)],
}],
encoding=[
    {"subword_models": ["bpe"],  "vocab_sizes": [2000, 4000]},
    {"subword_models": ["char"], "vocab_sizes": [200]},
],
```

Inspect what will run before committing GPU hours:

```python
for ds in builder.get_train_ds():
    print(ds.variant_id(as_path=True))
```

## Picking metrics

`PredictConfig(metrics=...)` takes a set of metric names, routed to the right backend
automatically:

| Metric string | Backend | Extra needed |
| --- | --- | --- |
| `"bleu"`, `"chrf"`, `"ter"` | sacreBLEU | — (base install) |
| `"bertscore"` | BERTScore | — (base install) |
| `"comet"` | COMET | downloads a model on first use |
| `"hg_<name>"` | HuggingFace `evaluate` | `pip install -e '.[hf]'` |

```python
PredictConfig(metrics={"bleu", "chrf", "comet"}, beams=[1, 5])
```

`beams` is a list, so you can score greedy (`beam1`) and beam search (`beam5`) in one
call; each ends up as its own column in the report.

## Choosing what to evaluate on

`eval_mode` decides which test sets each trained model is scored against:

- `"same"` — only the cell's own test set.
- `"compatible"` — every test set with the same language pair (best for cross-corpus
  generalization).
- `"all"` — everything.

See the [eval_mode table](../concepts/grid.md#choosing-what-to-evaluate-eval_mode).

## Compare the cells

Collect every `predict()` result and hand the list to `generate_report`, then plot:

```python
from autonmt.reporting.report import generate_report
from autonmt.reporting.figures import plot_model_comparison

df_report, df_summary = generate_report(scores=scores, output_path="outputs/sweep")
plot_model_comparison(
    df_report=df_report, out_dir="outputs/sweep/plots",
    metric="translations.beam5.sacrebleu_bleu_score",
    xlabel="Train variant", ylabel="BLEU", title="Vocab-size sweep",
)
```

→ More on outputs in [Reports & plots](reports.md).
