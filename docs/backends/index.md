# Backends

A **backend** is the engine behind a translator. All three implement the same
`fit()` / `predict()` contract from
[`BaseTranslator`][autonmt.backends._base.translation_engine.BaseTranslator], so the
experiment code around them is identical - you switch engines by changing one class.

```{.text .pipeline}
                BaseTranslator   (fit / predict, shared pipeline)
                      │
      ┌───────────────┼────────────────────┐
      ▼               ▼                     ▼
AutonmtTranslator  HuggingFaceTranslator  FairseqTranslator
(Lightning models)  (transformers seq2seq)  (fairseq CLI, deprecated)
```

| Backend                                   | Engine            | Toolkit folder        | Status          |
| ----------------------------------------- | ----------------- | --------------------- | --------------- |
| [`AutonmtTranslator`](autonmt.md)         | PyTorch Lightning | `models/autonmt/`     | **Recommended** |
| [`HuggingFaceTranslator`](huggingface.md) | `transformers`    | `models/huggingface/` | Stable          |
| [`FairseqTranslator`](fairseq.md)         | Fairseq CLI       | `models/fairseq/`     | ⚠️ Deprecated   |

## What the base class does

`BaseTranslator` owns the parts every backend shares:

- the public `fit()` / `predict()` surface and config merging,
- selecting which test sets to evaluate via `filter_eval_datasets` / `eval_mode`,
- routing metrics through the [`MetricBackend`](../reference/evaluation.md) registry,
- assembling the per-run report dict.

Subclasses implement the abstract hooks `_train`, `_translate`, `_get_lang_pair`, and
`_get_run_metadata`.

## SPM round-trip vs direct mode

SentencePiece-based backends (AutoNMT, Fairseq) encode the evaluation splits with the same
subword model the training run used, then decode the hypotheses back. That round-trip lives
in [`spm_pipeline.py`](https://github.com/salvacarrion/autonmt/blob/main/autonmt/backends/_base/spm_pipeline.py)
and is wired by composition: a backend sets `self._spm = SPMTranslatePipeline(...)` in its
constructor, and `translate()` branches on whether it's set.

The HuggingFace backend brings its **own** tokenizer, so it leaves `_spm = None` and writes
`src.txt` / `ref.txt` / `hyp.txt` directly - _direct mode_. From your point of view nothing
changes; the score schema is identical.

## Mixing backends in one report

Because every backend emits the same flattened score schema, you can drop an AutoNMT model,
a fine-tuned HuggingFace checkpoint, and a Fairseq baseline into a **single**
`generate_report` call and compare them in one table. See [Reports & plots](../guides/reports.md).
