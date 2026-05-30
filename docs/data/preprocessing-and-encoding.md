# Preprocessing & subword encoding

Between "raw parallel text" and "tensors a model can train on" sit two distinct jobs, and
AutoNMT keeps them deliberately separate:

- **Preprocessing** — *subword-agnostic* text cleanup: normalize, filter by length, dedupe,
  shuffle. The output is the same whether you'll encode with `word`, `bpe`, or `bytes`.
- **Encoding** — *subword-dependent* tokenization: turn clean text into subword units with a
  learned (or rule-based) tokenizer.

The split matters because it's why a grid over `subword_models × vocab_sizes` is cheap:
preprocessing runs **once** per split (`2_preprocessed/`), and only the encoding stage
(`4_encoded/<sw>/<vs>/`) re-runs per subword/vocab combination.

## Preprocessing (the `2_preprocessed` stage)

`autonmt.datasets.preprocessing` provides composable cleanup functions. You rarely call them
through the builder's [hooks](#hooks); the building blocks are:

| Function | What it does |
| --- | --- |
| `normalize_lines(lines, seq=...)` | Unicode normalization (default `NFKC` + `Strip`) via `tokenizers.normalizers` |
| `preprocess_pairs(src, tgt, ...)` | Paired cleanup: normalize, length filter, length-ratio filter, dedupe, shuffle — keeping src/tgt aligned |
| `preprocess_lines(lines, ...)` | Single-side cleanup (used at predict time on the source only) |

`preprocess_pairs` is the workhorse for training data. Its knobs:

```python
preprocess_pairs(
    src_lines, tgt_lines,
    normalize_fn=normalize,        # e.g. NFKC + Strip
    min_len=1, max_len=None,       # drop empty / overly long sentences
    max_len_percentile=None,       # or cap length at a percentile of the corpus
    remove_duplicates=True,        # drop exact duplicate pairs
    max_len_ratio_percentile=99,   # drop pairs whose src/tgt lengths differ wildly
    safe_len_ratio=2.0,            # …but never below this ratio threshold
    shuffle_lines=False,
)
```

!!! info "Why filter by length ratio?"
    A source sentence of 5 tokens aligned to a target of 60 is almost always a misalignment
    or a bad scrape, not a real translation. Such pairs add noise and waste capacity. The
    ratio filter drops pairs where `max(len)/min(len)` exceeds a data-driven threshold (the
    `max_len_ratio_percentile`), but never goes below `safe_len_ratio` so legitimate
    variation survives.

## Hooks: injecting your own cleanup { #hooks }

You don't subclass anything to customize preprocessing — you pass **callable hooks** to the
builder. There are three, fired at different stages:

| Hook | Fired on | Signature |
| --- | --- | --- |
| `preprocess_raw_fn` | the raw files, before splitting | `fn(data, ds) -> (src_lines, tgt_lines)` |
| `preprocess_splits_fn` | each split, after splitting | `fn(data, ds) -> (src_lines, tgt_lines)` |
| `preprocess_fn` *(predict-time)* | the source at translation time | `fn(data, ds) -> lines` |

For the paired hooks, `data` carries `data["src"]["lines"]` and `data["tgt"]["lines"]`; for
the predict-time hook, `data["lines"]` and `data["lang"]`. A typical setup wires the same
normalization through all of them:

```python
from tokenizers.normalizers import NFKC, Strip
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.preprocessing import normalize_lines, preprocess_pairs, preprocess_lines

def normalize(lines):
    return normalize_lines(lines, seq=[NFKC(), Strip()])

def clean_pairs(data, ds):                 # for preprocess_raw_fn / preprocess_splits_fn
    return preprocess_pairs(
        data["src"]["lines"], data["tgt"]["lines"],
        normalize_fn=normalize, remove_duplicates=True, max_len_ratio_percentile=99,
    )

def clean_source(data, ds):                # for the predict-time preprocess_fn
    return preprocess_lines(data["lines"], normalize_fn=normalize)

builder = DatasetBuilder(
    base_path="data", datasets=[...], encoding=[...],
    preprocess_raw_fn=clean_pairs,
    preprocess_splits_fn=clean_pairs,
).build()
```

The predict-time `preprocess_fn` is passed to [`predict`](../toolkit/predict.md) instead, so
your test source gets the *same* normalization the training data saw — otherwise you'd
measure a train/test preprocessing mismatch rather than model quality.

!!! warning "Match train and predict preprocessing"
    Whatever normalization you apply to training data, apply the equivalent to the
    predict-time source via `preprocess_fn`. A capitalization or Unicode-form mismatch between
    the two silently depresses scores.

## Encoding (the `4_encoded` stage)

Encoding turns clean text into subword units. AutoNMT supports six schemes plus an
orthogonal byte-fallback flag. The choice drives both the tokenizer and the on-disk path
(`4_encoded/<subword>/<vocab>/`).

### The subword models

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

### `word` is special

Choosing `word` flips the `pretok_flag`: AutoNMT first runs **Moses pretokenization**
(splitting punctuation, handling contractions) into `3_pretokenized/`, and for `word` runs
that pretokenized text *is* the encoded output. This is also why decoding a `word` run runs
Moses *de*tokenization to restore natural spacing.

### How encoding is wired

You don't usually call the encoders directly — the builder does, via
`autonmt.datasets.encoding`:

- `pretokenize_file(...)` — Moses tokenize (for `word`).
- `encode_file(...)` — SentencePiece-encode (word/char/bpe/unigram), hex-encode (`bytes`),
  or copy (`none`).
- `decode_file(...)` / `decode_lines(...)` — the inverse, used after translation to turn
  model output back into readable text (SPM decode, then Moses detokenize for `word`).

The encode/decode round-trip during evaluation is handled automatically by the
[SPM pipeline](../architecture/toolkit-abstraction.md#spm-pipeline-mode-autonmt-fairseq) in
the AutoNMT and Fairseq backends — you only pick the `subword_model`; AutoNMT does the rest.

---

Next: the artifact encoding produces and your model consumes —
**[Vocabularies](vocabularies.md)**.
