# Choosing a backend

A **backend** is the NMT toolkit that runs underneath `fit` / `predict`. AutoNMT ships
three, all implementing the same [`BaseTranslator`
contract](../architecture/toolkit-abstraction.md), so switching is a one-line change while
your data prep, scoring, and reports stay identical.

| Backend | Class | Use it when… | Install |
| --- | --- | --- | --- |
| **AutoNMT** (Lightning) | `AutonmtTranslator` | You're training a **custom architecture from scratch** and want full control over the model, decoding, and data pipeline | core |
| **HuggingFace** | `HuggingFaceTranslator` | You want to **fine-tune or evaluate a pretrained** seq2seq checkpoint (Marian, mBART, NLLB, T5…) | `[hf-models]` |
| **Fairseq** *(deprecated)* | `FairseqTranslator` | You need to **reproduce an existing Fairseq baseline** | `[fairseq]` |

## The one-line swap

The same script, three engines — only the translator object changes:

```python
# Native Lightning engine (custom Transformer)
trainer = AutonmtTranslator.from_dataset(
    train_ds, model=Transformer.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab, run_prefix="exp")

# Fine-tune a pretrained HuggingFace model
trainer = HuggingFaceTranslator.from_dataset(
    train_ds, model_id="Helsinki-NLP/opus-mt-de-en", run_prefix="exp")

# Reproduce a Fairseq baseline (deprecated)
trainer = FairseqTranslator.from_dataset(
    train_ds, src_vocab=src_vocab, tgt_vocab=tgt_vocab, run_prefix="exp")
```

After that line, `trainer.fit(...)` and `trainer.predict(...)` are called the same way, and
`generate_report(...)` consumes the results identically. That's the [Keras-style
abstraction](../introduction/philosophy.md#keras) in practice.

## How they differ underneath

| | AutoNMT | HuggingFace | Fairseq |
| --- | --- | --- | --- |
| Model source | your `LitSeq2Seq` | pretrained `AutoModelForSeq2SeqLM` | Fairseq CLI archs |
| Training | PyTorch Lightning | `Seq2SeqTrainer` (fine-tune) | `fairseq-train` (subprocess) |
| Tokenization | dataset's SentencePiece | the model's own HF tokenizer | dataset's SentencePiece |
| Translate mode | [SPM pipeline](../architecture/toolkit-abstraction.md#spm-pipeline-mode-autonmt-fairseq) | [direct](../architecture/toolkit-abstraction.md#direct-mode-huggingface) | SPM pipeline |
| Decoding | AutoNMT's [search strategies](../toolkit/decoding.md) | `model.generate` | Fairseq's generator |
| Maintained | ✅ actively | ✅ | ❌ archived 2026-03-20 |

The practical consequences:

- **AutoNMT** is the only backend where you control the architecture and the decoding
  algorithm. It's documented in depth in [The AutoNMT toolkit](../toolkit/overview.md) — this
  section doesn't repeat it.
- **HuggingFace** brings its own tokenizer, so the dataset's subword model is irrelevant to
  it (it reads the preprocessed splits and tokenizes itself). Best when you want a strong
  pretrained starting point. → [HuggingFace](huggingface.md)
- **Fairseq** still works but is deprecated; prefer AutoNMT for new work. → [Fairseq](fairseq.md)

## What's shared no matter the backend

Because all three honor the same contract, you always get — for free, identically:

- the same [config persistence + environment snapshot](../architecture/layout-and-reproducibility.md),
- the same [`eval_mode`](../toolkit/predict.md#eval-mode) test-set filtering,
- the same [metric backends](../evaluation/metrics.md) and [report schema](../evaluation/reports.md),
- the same on-disk run layout (only the `<toolkit>` folder name differs:
  `models/autonmt/…`, `models/huggingface/…`, `models/fairseq/…`).

!!! tip "Mix backends in one report"
    Because every backend emits the same flattened score schema, you can drop an AutoNMT
    model, a fine-tuned HuggingFace checkpoint, and a Fairseq baseline into a **single**
    `generate_report` call and compare them in one table.

To document a *new* toolkit as a backend, see [Extending
AutoNMT](../extending/index.md#a-custom-backend).
