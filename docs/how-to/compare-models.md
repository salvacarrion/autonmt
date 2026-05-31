# Compare multiple models

To compare architectures fairly, hold the data fixed and vary only the model. Because every
run is scored identically and collected into one table, the comparison is controlled by
construction.

```python
from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.nn.models import Transformer, BahdanauRNN, ConvS2S
from autonmt.reporting.report import Report

# One prepared dataset cell, reused by every model.
train_ds = builder.get_train_ds()[0]
test_ds  = builder.get_test_ds()
src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)

# The architectures under test — each gets its own run_name, so artifacts don't collide.
models = {
    "transformer": Transformer.from_vocabs(src_vocab, tgt_vocab),
    "bahdanau":    BahdanauRNN.from_vocabs(src_vocab, tgt_vocab),
    "convs2s":     ConvS2S.from_vocabs(src_vocab, tgt_vocab),
}

all_scores = []
for name, model in models.items():
    trainer = AutonmtTranslator.from_dataset(
        train_ds, model=model,
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        run_prefix=name,                       # transformer_*, bahdanau_*, convs2s_*
    )
    trainer.fit(train_ds, config=FitConfig(max_epochs=10, batch_size=128, seed=42))
    all_scores.append(trainer.predict(test_ds, config=PredictConfig(beams=[5], metrics={"bleu", "chrf"})))

report = (
    Report.from_runs(all_scores, output_path="outputs/arch-comparison")
    .save()
    .plot_comparison("bleu", beam=5)        # grouped bar chart across models
)
print(report)
```

The report's `model__architecture` column labels each row, and the plot puts the three
systems side by side. See the [Model catalog](../guide/models/catalog.md) for the
architectures and [Reports & plots](../guide/evaluation/reports.md) for the output schema.

!!! tip "Same idea, any axis"
    Swap "vary the model" for "vary the tokenization" (add `subword_models` /`vocab_sizes` to
    the builder) or "vary the data size" (add `sizes`) and the loop is unchanged — the grid
    declaration carries the comparison. For a single-axis sweep with a line plot, use
    `report.plot_sweep(...)` ([Reports & plots](../guide/evaluation/reports.md#metric-sweeps)).

!!! warning "Comparing fairly means controlling noise"
    A single run per model conflates architecture differences with training noise. For a
    claim, run a few [seeds and a significance test](reproduce.md).
