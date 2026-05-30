# Preprocessing & vocabularies

This guide covers the two cleanup hooks AutoNMT gives you and how to choose a subword
model. Mirrors
[`examples/basics/03_preprocessing_and_vocabs.py`](https://github.com/salvacarrion/autonmt/blob/main/examples/basics/03_preprocessing_and_vocabs.py).

## Where preprocessing runs

```text
raw files  ──preprocess_raw_fn──►  splits  ──preprocess_splits_fn──►  encoded
                                      │
                                      └─ at predict() time the test source is run
                                         through preprocess_fn (PredictConfig) before encoding
```

There are **three** hooks, and the distinction matters:

| Hook                   | Passed to        | Runs                               | Use for                                                              |
| ---------------------- | ---------------- | ---------------------------------- | -------------------------------------------------------------------- |
| `preprocess_raw_fn`    | `DatasetBuilder` | **once**, on the raw corpus        | opinionated, one-time decisions: dedupe, shuffle, percentile filters |
| `preprocess_splits_fn` | `DatasetBuilder` | per split (train/val/test)         | normalization you want applied identically to every split            |
| `preprocess_fn`        | `PredictConfig`  | on the test source at predict time | the same normalization, so inference matches training                |

The golden rule: **dedupe/shuffle/filter belong in `preprocess_raw_fn`**; per-line
normalization belongs in `preprocess_splits_fn` (and must be mirrored in `preprocess_fn`).
Don't dedupe or shuffle per-split - that would treat val/test differently from train.

## Writing the hooks

A hook receives `(data, ds)` and returns the cleaned lines. The helpers in
[`autonmt.datasets.preprocessing`](../reference/datasets.md) do the heavy lifting:

```python
from tokenizers.normalizers import NFKC, Strip, Lowercase
from autonmt.datasets.preprocessing import normalize_lines, preprocess_lines, preprocess_pairs

def normalize(lines):
    # Any tokenizers.normalizers.* sequence works here.
    return normalize_lines(lines, seq=[NFKC(), Strip(), Lowercase()])

def preprocess_raw(data, ds):
    # One-time aggressive cleaning of the raw corpus.
    return preprocess_pairs(
        data["src"]["lines"], data["tgt"]["lines"],
        normalize_fn=normalize,
        min_len=1,                     # drop empty lines
        max_len_percentile=99,         # drop the longest 1% (likely noise)
        remove_duplicates=True,        # drop exact-duplicate pairs
        max_len_ratio_percentile=99,   # drop pairs with extreme src/tgt length ratio
        shuffle_lines=True,            # shuffle once so file order doesn't bias splits
    )

def preprocess_splits(data, ds):
    # Per-split: normalize only. No dedupe/shuffle here.
    return preprocess_pairs(data["src"]["lines"], data["tgt"]["lines"], normalize_fn=normalize)

def preprocess_predict(data, ds):
    # Mirror training normalization on the test source.
    return preprocess_lines(data["lines"], normalize_fn=normalize)
```

`preprocess_pairs` works on aligned source/target lists (so it can apply pair-level filters
like length ratio); `preprocess_lines` works on a single list (the predict-time source,
where there's no target to align against).

## Choosing a subword model

Declared per encoding entry via `subword_models`:

| Model     | What it does                           | Notes                                                        |
| --------- | -------------------------------------- | ------------------------------------------------------------ |
| `bpe`     | Classic BPE merges (SentencePiece)     | A solid default                                              |
| `unigram` | SentencePiece unigram LM               | Often slightly better than BPE                               |
| `word`    | Whitespace + **Moses** pretokenization | Large vocab, fast model; triggers the `3_pretokenized` stage |
| `char`    | One symbol per character               | Tiny vocab, long sequences                                   |
| `bytes`   | Pure byte level                        | Vocab is always 256; **ignores `vocab_sizes`**               |

### Byte fallback: the `+bytes` shorthand

Suffix any model with `+bytes` to enable SentencePiece byte fallback - unseen characters
are emitted as raw bytes instead of `<unk>`. Invaluable for rare scripts, emoji, and names.

```python
encoding=[{"subword_models": ["unigram+bytes"], "vocab_sizes": [4000]}]
# equivalent to:
encoding=[{"subword_models": ["unigram"], "vocab_sizes": [4000], "byte_fallback": True}]
```

Because the flag is orthogonal to the model, you compare _with vs without_ fallback by
declaring two entries - they live at different on-disk paths (`unigram/4000` vs
`unigram+bytes/4000`). See [The grid](../concepts/grid.md#subword-models-and-the-bytes-shorthand).

## Inspecting the artifacts

After `build()`, the trained SPM model and the tab-separated frequency vocab land under the
cell's `vocabs/` folder. Open them to debug a subword choice:

```python
print(train_ds.get_vocab_path())     # vocabs/<subword>/<vocab>/
print(train_ds.get_encoded_path())   # data/4_encoded/<subword>/<vocab>/
```

Pass `verbose=True` to `build()` to print per-split token statistics as it runs.

## From vocab to model

`build_vocabs` loads the trained vocab into the `Vocabulary` objects the model needs.
`max_tokens` caps the vocabulary size loaded into the model (handy for quick iteration):

```python
src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)
model = Transformer.from_vocabs(src_vocab, tgt_vocab)
```

Set `merge_vocabs=True` on the builder to share a single vocabulary across source and
target (common for related languages or shared scripts).
