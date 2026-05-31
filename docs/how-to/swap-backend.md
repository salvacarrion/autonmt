# Run with another backend

Because all three backends honor the same [`BaseTranslator` contract](../guide/backends/choosing.md),
moving an experiment to a different toolkit is a **one-line change** — the data prep,
scoring, and report are identical.

## Swap the translator

```python
# Native Lightning engine (custom Transformer)
trainer = AutonmtTranslator.from_dataset(
    train_ds, model=Transformer.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab, run_prefix="exp")

# Fine-tune / evaluate a pretrained HuggingFace checkpoint
trainer = HuggingFaceTranslator.from_dataset(
    train_ds, model_id="Helsinki-NLP/opus-mt-de-en", run_prefix="exp")

# Reproduce a Fairseq baseline (deprecated)
trainer = FairseqTranslator.from_dataset(
    train_ds, src_vocab=src_vocab, tgt_vocab=tgt_vocab, run_prefix="exp")
```

After that line, `fit` / `predict` / `Report` are called the same way. Each backend
writes to its own `models/<toolkit>/` run folder, so they never collide.

## Put them in one report

The real payoff: because every backend emits the **same flattened score schema**, you can
compare a from-scratch model, a fine-tuned pretrained checkpoint, and a baseline in a single
table.

```python
from autonmt.reporting.report import Report

all_scores = []

# A — your native Transformer
a = AutonmtTranslator.from_dataset(train_ds, model=Transformer.from_vocabs(src_vocab, tgt_vocab),
                                   src_vocab=src_vocab, tgt_vocab=tgt_vocab, run_prefix="scratch")
a.fit(train_ds, config=FitConfig(max_epochs=10, seed=42))
all_scores.append(a.predict(test_ds, config=PredictConfig(beams=[5], metrics={"bleu", "chrf"})))

# B — a pretrained HuggingFace baseline (no fit)
b = HuggingFaceTranslator.from_dataset(train_ds, model_id="Helsinki-NLP/opus-mt-de-en", run_prefix="opus")
all_scores.append(b.predict(test_ds, config=PredictConfig(beams=[5], metrics={"bleu", "chrf"},
                                                          eval_mode="compatible")))

report = Report.from_runs(all_scores, output_path="outputs/backend-bakeoff").save()
print(report)   # engine column distinguishes the rows
```

The report's `engine` column (`autonmt` / `huggingface` / `fairseq`) labels each row. See
[Choosing a backend](../guide/backends/choosing.md) for what differs underneath, and
[HuggingFace](../guide/backends/huggingface.md) / [Fairseq](../guide/backends/fairseq.md) for
backend-specific knobs.
