# Vocabularies

A **vocabulary** is the bridge between text and tensors: it maps subword pieces to integer
ids (to feed the model) and back (to read its output). In AutoNMT a vocabulary is a small,
explicit object you build from a prepared dataset and hand to both the model and the
translator.

## The classes

| Class | Role |
| --- | --- |
| [`BaseVocabulary`](../../reference/vocabularies.md) | Abstract base: the four special tokens + `encode`/`decode` contract |
| [`Vocabulary`](../../reference/vocabularies.md) | The concrete whitespace-backed vocabulary used everywhere |
| `vocab_builder` | Creates the on-disk vocab artifacts during `DatasetBuilder.build()` |

Every vocabulary shares **four special tokens** with fixed default ids:

| Token | id | Meaning |
| --- | --- | --- |
| `<unk>` | 0 | unknown / out-of-vocabulary piece |
| `<s>` | 1 | start of sequence (`sos`) |
| `</s>` | 2 | end of sequence (`eos`) |
| `<pad>` | 3 | padding (fills short sequences in a batch) |

These ids are part of the contract — models read `vocab.pad_id`, decoders read
`vocab.sos_id` / `vocab.eos_id`, and so on, without hard-coding numbers.

!!! info "Why special tokens?"
    A model needs to know where a sentence **begins** and **ends** (it generates `</s>` to
    stop), how to **pad** variable-length sentences into a rectangular batch without the
    padding affecting the result, and what to emit when it hits something it can't represent
    (`<unk>`). Decoding strips `<s>`/`</s>` and ignores `<pad>` so you get clean text back.

## Building vocabularies

The artifacts (`.model` / `.vocab` / `.vocabf` files) are produced during
`DatasetBuilder.build()`. You then *load* them for a cell with one call:

```python
src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)
```

- For SentencePiece models (`word`/`bpe`/`unigram`/`char`), `build_vocabs` loads the trained
  SPM model plus its `.vocab`.
- For `bytes`, there's a fixed pseudo-vocabulary of hex byte tokens (no learned model).
- `max_tokens` caps how many tokens a single sentence contributes when batches are encoded
  later (long sentences are truncated, reserving room for `<s>`/`</s>`).

### Separate vs shared (merged) vocabularies

By default, source and target get **separate** vocabularies. Set `merge_vocabs=True` on the
builder to learn a **single shared** vocabulary over both languages:

```python
DatasetBuilder(..., merge_vocabs=True)
```

```python
src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)
# with merge_vocabs=True, src_vocab is tgt_vocab (the same shared instance)
```

!!! info "When to share a vocabulary"
    A **shared** vocabulary is the norm when source and target share an alphabet/script
    (most European pairs) — it enables tying input/output embeddings and helps related
    languages reuse subwords. **Separate** vocabularies make sense when the scripts differ
    (e.g. zh→en) so neither language wastes entries on the other's characters. It's a
    legitimate grid axis when you're unsure.

    When sharing, the model can also **tie embeddings** (share the decoder input embedding
    with the output projection) via `Transformer(..., tie_embeddings=True)` — see
    [Models](../models/using-a-model.md).

## Encoding and decoding

A `Vocabulary` is also a runtime encoder/decoder. You rarely call it directly — the
[`TranslationDataset`](../training/bucketing.md) encodes batches and the
[decoder](../translation/decoding.md) decodes outputs — but the surface is simple:

```python
ids = tgt_vocab.encode("a clean sentence")          # → [sos, .., eos] token ids
text = tgt_vocab.decode(ids, remove_special_tokens=True)  # → "a clean sentence"
```

- `encode` whitespace-splits the (already subword-segmented) text, maps pieces to ids,
  truncates to `max_tokens`, and wraps with `<s>`/`</s>`.
- `decode` strips special tokens and joins pieces back — or, for `bytes`, converts hex
  tokens back to characters.

The full SentencePiece/Moses round-trip (so `hyp.txt` reads as natural language, not
subword fragments) is applied automatically during evaluation by the backend's SPM pipeline.

## Inspecting a vocabulary

```python
len(src_vocab)               # vocab size (includes the 4 special tokens)
src_vocab.subword_model      # SubwordModel.BPE
src_vocab.lang               # 'de'
src_vocab.pad_id             # 3
src_vocab.get_tokens()[:10]  # the first pieces, in id order
```

The `.vocabf` file alongside the vocab stores **token frequencies**, which the
[dataset diagnostics](../evaluation/reports.md#dataset-diagnostics) use to plot the vocab
distribution.

## Custom vocabularies

Because `Vocabulary` subclasses `BaseVocabulary`, you can build your own (a different
tokenization scheme, a domain lexicon) and pass it straight to the model and translator —
`build_vocabs` is a convenience, not a requirement. Implement the
[`BaseVocabulary`](../../reference/vocabularies.md) contract (the four special-token ids plus
`encode`/`decode`) and the rest of the framework treats it like any other vocabulary.

---

That's the input side complete. The prepared, encoded, vocab'd data now flows into a
**[model](../models/using-a-model.md)** and a **[backend](../backends/choosing.md)**.
