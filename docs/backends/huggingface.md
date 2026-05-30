# HuggingFace backend

[`HuggingFaceTranslator`][autonmt.backends.huggingface.translation_engine.HuggingFaceTranslator]
wraps `transformers.AutoModelForSeq2SeqLM` so you can **evaluate** a pretrained seq2seq
checkpoint on your test set, or **fine-tune** it on your splits - using the same
`fit()` / `predict()` flow as every other backend.

Install the extras:

```bash
pip install -e '.[hf,hf-models]'
```

Mirrors [`examples/basics/06_huggingface_baseline_and_finetune.py`](https://github.com/salvacarrion/autonmt/blob/main/examples/basics/06_huggingface_baseline_and_finetune.py).

## Its own tokenizer

The HF backend brings its own tokenizer, so the `subword_models` / `vocab_sizes` you
declared in `encoding=` are used only for the AutoNMT-side dataset _identity_ and stats -
HF ignores the SentencePiece model and tokenizes the source itself. It runs in _direct
mode_ (`_spm = None`), writing `src.txt` / `ref.txt` / `hyp.txt` straight to disk. The
score schema is identical to the other backends.

## Evaluate a pretrained baseline

No `fit()` - the model is already trained, so jump straight to `predict()`:

```python
from autonmt.backends import HuggingFaceTranslator
from autonmt.backends._base.config import PredictConfig

baseline = HuggingFaceTranslator.from_dataset(
    train_ds,
    model_id="Helsinki-NLP/opus-mt-de-en",
    run_prefix="opus-baseline",
    device="auto",
)
scores = baseline.predict(test_datasets, config=PredictConfig(
    metrics={"bleu", "chrf"}, beams=[5],
    preprocess_fn=preprocess_predict, eval_mode="compatible", batch_size=8,
))
```

## Fine-tune the same checkpoint

`FitConfig` fields map onto `transformers.Seq2SeqTrainingArguments`:

| `FitConfig`                                                                     | `Seq2SeqTrainingArguments`           |
| ------------------------------------------------------------------------------- | ------------------------------------ |
| `max_epochs`                                                                    | `num_train_epochs`                   |
| `batch_size`                                                                    | `per_device_{train,eval}_batch_size` |
| `learning_rate`, `weight_decay`, `gradient_clip_val`, `accumulate_grad_batches` | same-named args                      |
| `patience`                                                                      | `EarlyStoppingCallback`              |
| `seed`, `num_workers`                                                           | `seed`, `dataloader_num_workers`     |

```python
finetuner = HuggingFaceTranslator.from_dataset(
    train_ds, model_id="Helsinki-NLP/opus-mt-de-en",
    run_prefix="opus-finetuned", device="auto",
)
finetuner.fit(
    train_ds,
    config=FitConfig(
        max_epochs=1, batch_size=8, learning_rate=2e-5,
        weight_decay=0.01, gradient_clip_val=1.0,
        patience=2, num_workers=0, seed=42,
        save_best=True, monitor="eval_loss",
    ),
    hf_training_args={"label_smoothing_factor": 0.1},  # HF-only knobs, win on collision
)
finetuned_scores = finetuner.predict(test_datasets, config=pred_cfg)
```

!!! note "HF-only training knobs"
Anything specific to the HF `Trainer` (label smoothing, mixed precision, …) goes through
`hf_training_args=dict(...)` on `.fit()`. On collision with `FitConfig`, the explicit
`hf_training_args` value wins.

## Same report as everyone else

Because the schema matches, the baseline and the fine-tuned run drop into the same report
alongside your AutoNMT models - one row per `(run, test_ds)` pair:

```python
generate_report(scores=[baseline_scores, finetuned_scores], output_path="outputs/hf")
```

## API reference

See [`HuggingFaceTranslator`][autonmt.backends.huggingface.translation_engine.HuggingFaceTranslator]
in the [backends API reference](../reference/backends.md#huggingfacetranslator).
