# HuggingFace

[`HuggingFaceTranslator`](../reference/backends.md) lets you **evaluate** any pretrained
seq2seq checkpoint on your test set, or **fine-tune** it on your splits — through the same
`fit` / `predict` flow as every other backend. Reach for it when you'd rather start from a
strong pretrained model (Marian/OPUS, mBART, NLLB, T5…) than train from scratch.

```bash
pip install -e '.[hf-models]'    # transformers + accelerate
```

## How it differs from the native engine

The HuggingFace backend brings its **own tokenizer**, so it runs in [direct
mode](../architecture/toolkit-abstraction.md#direct-mode-huggingface) (`_spm = None`): it
reads the preprocessed splits, tokenizes the source itself, calls `model.generate`, and
writes `src.txt` / `ref.txt` / `hyp.txt` straight to disk. Two implications:

- The `subword_models` / `vocab_sizes` you declared in `encoding=` are used only for the
  AutoNMT-side dataset **identity and stats** — HF ignores the SentencePiece model entirely.
- Decoding uses HF's `generate`, not AutoNMT's [search strategies](../toolkit/decoding.md).
  The `beams` you pass become `num_beams`; `max_len_a`/`max_len_b` become `max_new_tokens`.

The output artifacts and the [report schema](../evaluation/reports.md) are identical to the
other backends, so the results sit in the same table.

## Construct it

```python
from autonmt.backends import HuggingFaceTranslator

trainer = HuggingFaceTranslator.from_dataset(
    train_ds,
    model_id="Helsinki-NLP/opus-mt-de-en",   # Hub id or local checkpoint dir
    run_prefix="opus",
    device="auto",                           # auto / cuda / mps / cpu
    # tokenizer_id="...",                    # defaults to model_id
    # generation_kwargs={"no_repeat_ngram_size": 3},   # forwarded to model.generate
)
```

`from_dataset` auto-fills `src_lang` / `tgt_lang` from the dataset (used for
[`eval_mode`](../toolkit/predict.md#eval-mode) filtering and target-language metrics). The
model and tokenizer load **lazily** on first use, so you can construct the translator and
inspect paths without paying the download cost.

| Argument | Meaning |
| --- | --- |
| `model_id` | Hub id (e.g. `"facebook/nllb-200-distilled-600M"`) or a local checkpoint directory |
| `tokenizer_id` | Defaults to `model_id`; override if the tokenizer lives elsewhere |
| `device` | `"auto"` (CUDA → MPS → CPU), or pin it |
| `generation_kwargs` | Extra kwargs for `model.generate` (AutoNMT manages `num_beams` / `max_new_tokens`) |

## Evaluate a pretrained baseline

No `fit` — the model is already trained, so go straight to `predict`:

```python
from autonmt.backends._base.config import PredictConfig

scores = trainer.predict(test_datasets, config=PredictConfig(
    metrics={"bleu", "chrf"}, beams=[5],
    eval_mode="compatible", batch_size=8,
    preprocess_fn=clean_source,   # match the normalization your test corpus needs
))
```

This is the quickest way to get a strong baseline number on your data.

## Fine-tune the same checkpoint

`fit` fine-tunes via `transformers.Seq2SeqTrainer`. AutoNMT maps
[`FitConfig`](../toolkit/training.md#fitconfig) onto `Seq2SeqTrainingArguments`:

| `FitConfig` | `Seq2SeqTrainingArguments` |
| --- | --- |
| `max_epochs` | `num_train_epochs` |
| `batch_size` | `per_device_{train,eval}_batch_size` |
| `learning_rate`, `weight_decay` | same-named |
| `gradient_clip_val` | `max_grad_norm` |
| `accumulate_grad_batches` | `gradient_accumulation_steps` |
| `patience` | `EarlyStoppingCallback` |
| `seed`, `num_workers` | `seed`, `dataloader_num_workers` |
| `monitor` | `metric_for_best_model` |

```python
from autonmt.backends._base.config import FitConfig

trainer.fit(
    train_ds,
    config=FitConfig(
        max_epochs=3, batch_size=8, learning_rate=2e-5,
        weight_decay=0.01, gradient_clip_val=1.0,
        patience=2, seed=42, save_best=True, monitor="eval_loss",
    ),
    hf_training_args={"label_smoothing_factor": 0.1, "fp16": True},  # HF-only knobs
)
scores = trainer.predict(test_datasets, config=PredictConfig(metrics={"bleu"}, beams=[5]))
```

After training, the best model + tokenizer are saved to the run's `checkpoints/` folder and
`model_id` / `tokenizer_id` are repointed there, so the subsequent `predict` evaluates the
**fine-tuned** weights. Re-running skips training if a fine-tuned checkpoint already exists
(unless `force_overwrite=True`).

!!! note "HF-only training knobs win on collision"
    Anything specific to the HF `Trainer` (label smoothing, mixed precision, custom schedulers)
    goes through `hf_training_args=dict(...)`. On collision with a `FitConfig`-derived value,
    the explicit `hf_training_args` value wins — same "extras override" rule the
    [contract](../architecture/toolkit-abstraction.md) uses everywhere.

## HuggingFace metrics (`hg_*`)

Beyond AutoNMT's built-in metrics, you can request **any** metric from the HuggingFace
`evaluate` hub by prefixing it `hg_`:

```python
trainer.predict(test_datasets, config=PredictConfig(metrics={"bleu", "hg_meteor", "hg_ter"}))
```

`hg_meteor` loads the `meteor` metric via `evaluate`. Requires the `[hf]` extra. See
[Metrics](../evaluation/metrics.md#huggingface-metrics).

## Parameter-efficient fine-tuning (LoRA, adapters)

`HuggingFaceTranslator` wraps a standard `AutoModelForSeq2SeqLM`, so parameter-efficient
techniques like **LoRA** apply to the underlying model the usual way (e.g. wrapping it with
`peft` before training). AutoNMT doesn't add a built-in flag for it — that's a deliberate
[minimal-core](../introduction/philosophy.md#extensible) choice — but the runnable
[`examples/advanced/`](https://github.com/salvacarrion/autonmt/tree/main/examples/advanced)
scripts show LoRA, catastrophic-forgetting, and model-merging recipes on top of this backend.

---

The other engine you might swap in is the deprecated **[Fairseq](fairseq.md)** backend.
