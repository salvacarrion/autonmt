# Drive the pipeline manually

Everything in the User guide is a **convenience layer**. `AutonmtTranslator.from_dataset`,
`ds.build_vocabs`, `Transformer.from_vocabs`, `FitConfig`, `trainer.predict`,
`Report` — each wraps a few lower-level steps. This recipe unpacks those shortcuts
so you can drive the engine piece by piece, which is what you need when you want to:

- swap one component (custom decoder, custom model dims, custom callbacks),
- split `predict` to re-score with a new metric **without re-decoding**,
- assemble the report yourself (extra columns, merge with another experiment),
- run multi-seed experiments for publication-grade variance.

There are no new features here — it's the *same* pipeline, unpacked.

## Manual vocabularies

Shortcut → `src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)`.

```python
from autonmt.vocabularies import Vocabulary

src_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.src_lang)
tgt_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.tgt_lang)
```

## Manual model construction

Shortcut → `Transformer.from_vocabs(src_vocab, tgt_vocab)`.

```python
from autonmt.core.nn.models import Transformer

model = Transformer(
    src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
    padding_idx=src_vocab.pad_id,
    encoder_embed_dim=512, decoder_embed_dim=512,
    encoder_layers=6, decoder_layers=6,
    encoder_attention_heads=8, decoder_attention_heads=8,
    encoder_ffn_embed_dim=2048, decoder_ffn_embed_dim=2048,
    dropout=0.1, max_src_positions=1024, max_tgt_positions=1024,
)
```

## Manual translator construction

Shortcut → `AutonmtTranslator.from_dataset(train_ds, ..., run_prefix="exp")`.

Compute the run location and name yourself — this is where you tag a run with a custom name
or override the directory scheme:

```python
from autonmt.backends import AutonmtTranslator

runs_dir = train_ds.get_runs_path(toolkit="autonmt")
run_name = train_ds.get_run_name(run_prefix="ablation")   # or any string, e.g. "ablation-v2-seed42"

trainer = AutonmtTranslator(
    model=model,
    src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    runs_dir=runs_dir, run_name=run_name,
)
```

## `fit` with raw kwargs

Shortcut → `trainer.fit(train_ds, config=FitConfig(...))`.

The two styles are equivalent (precedence: defaults < `config=` < kwargs). Raw kwargs are
handy in a notebook; `FitConfig` is handy for type-checked, shareable configs. Toolkit extras
(`strategy`, `wandb_params`, `use_bucketing`) ride along either way:

```python
trainer.fit(
    train_ds,
    max_epochs=10, batch_size=128, learning_rate=1e-3,
    optimizer="adam", monitor="val_loss", save_best=True,
    seed=42, force_overwrite=False,
)
```

## Split the pipeline: translate → score → parse

Shortcut → `scores = trainer.predict(test_datasets, config=PredictConfig(...))`.

`predict` runs three public methods in sequence. Splitting them lets you re-score without
re-decoding, plug a per-call decoder, or inspect the intermediate
`src.txt` / `ref.txt` / `hyp.txt` artifacts:

```python
from autonmt.core.decoding import BeamSearch

metrics, beams = {"bleu", "chrf"}, [5]
decoder = BeamSearch(length_penalty=1.2)          # custom decoder

# filter_eval_datasets runs inside predict(); doing it here lets you log/skip/reorder.
eval_datasets = trainer.filter_eval_datasets(test_datasets, eval_mode="compatible")

scores = []   # one entry per eval_ds — matches predict()'s return shape
for eval_ds in eval_datasets:
    # 1. Decode (writes hyp.tok → hyp.txt, src.txt, ref.txt).
    trainer.translate(
        eval_ds, beams=beams, preprocess_fn=None, force_overwrite=False,
        max_len_a=1.2, max_len_b=50, batch_size=64, max_tokens=None,
        devices="auto", accelerator="auto", num_workers=0,
        checkpoint="best", decoder=decoder,
    )
    # 2. Score existing hypotheses. Re-run this alone to add a new metric — no re-decoding.
    trainer.score_translations(eval_ds, beams=beams, metrics=metrics, force_overwrite=False)

    # 3. Parse the per-metric files back into the report dict shape.
    scores.append(trainer.parse_metrics(eval_ds, beams=beams, metrics=metrics))
```

!!! tip "Re-scoring is cheap; re-decoding isn't"
    Decoding with beam search is the expensive step. Because `translate` caches `hyp.txt`, you
    can iterate on metrics by calling only `score_translations` + `parse_metrics` afterward.
    Add COMET to an already-decoded run for the price of the metric, not the decode.

## Build the report by hand

Shortcut → `Report.from_predict(scores, output_path=out).save().plot_comparison("bleu")`.

```python
from autonmt.utils import fileio
from autonmt.reporting.report import Report, scores_to_dataframe, summarize_scores

# `Report` wraps these transforms — reach for them to add columns or skip disk.
df_report = scores_to_dataframe([scores])      # == Report(...).df
df_summary = summarize_scores(df_report)       # == Report(...).summary

fileio.make_dir(f"{out}/reports")
fileio.save_json([scores], f"{out}/reports/report.json")
df_report.to_csv(f"{out}/reports/report.csv", index=False)

# Plotting recipes live on `Report`; ask by metric name (beam inferred).
Report.from_predict(scores, output_path=out).plot_comparison(
    "bleu", xlabel="Run", ylabel="BLEU", title="Manual run")
print(Report.from_predict(scores))             # tidy summary table
```

## Inspect checkpoints

```python
ckpt_path = trainer.get_checkpoint_path(mode="best")   # path without loading
trainer.load_checkpoint("best")                        # load best/last/filename/abs-path into the model
```

---

For multi-seed variance and significance testing, see
[Reproduce an experiment](reproduce.md). To run a *different* engine with the same script,
[Run with another backend](swap-backend.md).
