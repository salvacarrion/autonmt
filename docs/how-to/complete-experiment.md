# Run a complete experiment

A full AutoNMT run is four moves: **build the data → train → translate & score → report.**
This recipe is a self-contained script you can paste and adapt. (For a line-by-line
walkthrough of the same shape, see the [Quickstart](../get-started/quickstart.md); for the
concepts, [The experiment workflow](../guide/experiments/workflow.md).)

```python
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.nn.models import Transformer
from autonmt.reporting.report import Report

# 1. Data onto disk (skip if you already have train/val/test splits).
download_hf_dataset(
    hf_id="bentrevett/multi30k", base_path="data",
    dataset_name="multi30k", lang_pair="de-en", src_field="de", tgt_field="en",
)

builder = DatasetBuilder(
    base_path="data",
    datasets=[{"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]}],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [8000]}],
).build()

# 2–3. Train and score each cell of the grid (one cell here).
all_scores = []
for train_ds in builder.get_train_ds():
    src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)
    trainer = AutonmtTranslator.from_dataset(
        train_ds,
        model=Transformer.from_vocabs(src_vocab, tgt_vocab),
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        run_prefix="run",
    )
    trainer.fit(train_ds, config=FitConfig(max_epochs=10, batch_size=128,
                                           learning_rate=5e-4, scheduler="noam",
                                           warmup_steps=4000, patience=5, seed=42))
    all_scores.append(trainer.predict(
        builder.get_test_ds(),
        config=PredictConfig(beams=[5], metrics={"bleu", "chrf"}, eval_mode="same"),
    ))

# 4. One comparable report.
report = (
    Report.from_runs(all_scores, output_path="outputs/run")
    .save()
    .plot_comparison("bleu", beam=5)
)
print(report)
```

What you get: a prepared dataset cell under `data/`, a self-describing run under
`models/autonmt/runs/`, and `report.{json,csv}` + a plot under `outputs/run/` — all explained
in [Understanding the output](../get-started/understanding-the-output.md).

!!! tip "Re-runs are incremental"
    Run the script again and AutoNMT skips every stage already on disk — data prep, encoding,
    and even training if a checkpoint exists. To force one stage, delete its folder (or pass
    `force_overwrite=True`). See [Configuration & reproducibility](../guide/experiments/configuration.md).

Scale this from one cell to a real sweep by adding axes to `datasets` / `encoding` — the loop
body doesn't change. That's [Compare multiple models](compare-models.md) and the
[grid-first idea](../concepts/philosophy.md#grid-first).
