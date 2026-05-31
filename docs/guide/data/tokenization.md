# Subword tokenization

Once text is [cleaned](preprocessing.md), encoding turns it into the subword units a model
actually consumes. This is the *subword-dependent* half of data prep: the choice of scheme
drives both the tokenizer and the on-disk path (`4_encoded/<subword>/<vocab>/`).

!!! info "Subword tokenization, intuitively"
    A fixed vocabulary can't hold every word of a language, so anything unseen would map to
    `<unk>` and be lost. Subword tokenization fixes this by working with *reusable fragments*:

    - **BPE** starts from characters and greedily merges the most frequent adjacent pair over
      and over, building up common chunks. Frequent words end up as single units; rare words
      fall back to smaller pieces. No word is ever fully unknown.
    - **Unigram** instead starts from a large candidate set and *prunes* it, keeping the
      subset that best explains the corpus under a unigram language model. It can represent a
      string multiple ways and picks the most probable segmentation.
    - **char** / **bytes** go to the smallest units — every character or every UTF-8 byte — so
      the vocabulary is tiny and *nothing* is ever out-of-vocabulary, at the cost of longer
      sequences.

    Bigger vocab → shorter sequences but more parameters and more rare entries; smaller vocab
    → the opposite. Sweeping `vocab_sizes` is exactly how you find the trade-off for your data
    — which is why it's a grid axis.

## The subword models

AutoNMT supports six schemes plus an orthogonal byte-fallback flag:

| `subword_model` | What a token is | Vocab? | Backed by |
| --- | --- | --- | --- |
| `word` | a whitespace word (after Moses pretokenization) | yes | SentencePiece (+ Moses) |
| `char` | a single character | yes | SentencePiece |
| `bpe` | a Byte-Pair-Encoding merge unit | yes | SentencePiece |
| `unigram` | a Unigram-LM subword | yes | SentencePiece |
| `bytes` | a hex-encoded byte of UTF-8 | no (pseudo) | built-in |
| `none` | no encoding (passthrough) | no | — |

You can also append **`+bytes`** to a SentencePiece model (`"bpe+bytes"`,
`"unigram+bytes"`) to enable SentencePiece's **byte fallback**: any character the subword
model can't represent decomposes into raw bytes instead of becoming `<unk>`.

## `word` is special

Choosing `word` flips the `pretok_flag`: AutoNMT first runs **Moses pretokenization**
(splitting punctuation, handling contractions) into `3_pretokenized/`, and for `word` runs
that pretokenized text *is* the encoded output. This is also why decoding a `word` run runs
Moses *de*tokenization to restore natural spacing.

## How encoding is wired

You don't usually call the encoders directly — the builder does, via
`autonmt.datasets.encoding`:

- `pretokenize_file(...)` — Moses tokenize (for `word`).
- `encode_file(...)` — SentencePiece-encode (word/char/bpe/unigram), hex-encode (`bytes`),
  or copy (`none`).
- `decode_file(...)` / `decode_lines(...)` — the inverse, used after translation to turn
  model output back into readable text (SPM decode, then Moses detokenize for `word`).

The encode/decode round-trip during evaluation is handled automatically by the
[SPM pipeline](../backends/choosing.md#two-translate-modes) in the AutoNMT and Fairseq
backends — you only pick the `subword_model`; AutoNMT does the rest.

!!! tip "The low-level tokenizer engines"
    Moses and SentencePiece are wrapped in `autonmt.datasets.tokenizers`. You almost never
    touch them directly, but if you need to train or apply a tokenizer outside the builder,
    that's where the thin engines live.

---

Encoding produces the artifact your model indexes into — the **[Vocabulary](vocabularies.md)**.
