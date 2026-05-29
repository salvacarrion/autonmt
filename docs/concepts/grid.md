# The grid

The grid is AutoNMT's central idea. Instead of writing a script per configuration, you
declare the axes you want to vary and let the builder produce one fully-materialized
**cell** per combination.

## The axes

`DatasetBuilder` takes two declarative blocks:

```python
builder = DatasetBuilder(
    base_path="datasets/demo",
    datasets=[{
        "name": "multi30k",
        "languages": ["de-en"],                       # axis: language pairs
        "sizes": [("original", None), ("10k", 10_000)],  # axis: training sizes
    }],
    encoding=[
        {"subword_models": ["bpe"], "vocab_sizes": [2000, 4000]},  # axis: subword × vocab
        {"subword_models": ["char"], "vocab_sizes": [200]},
    ],
).build()
```

This declares:

| Axis | Values | Count |
| --- | --- | --- |
| Language pairs | `de-en` | 1 |
| Training sizes | `original`, `10k` | 2 |
| Subword × vocab | `bpe/2000`, `bpe/4000`, `char/200` | 3 |

The cross-product is `1 × 2 × 3 = 6` dataset cells. `build()` materializes all six on
disk; `get_train_ds()` returns the six `Dataset` objects.

## Sizes: truncated variants of the same corpus

Each entry in `sizes` is a `(label, n_lines)` tuple. AutoNMT writes a variant truncated to
the first `n_lines` of the training split, stored at its own path so variants never
collide. The reserved label **`"original"`** is the un-truncated reference and ignores any
line cap (`None`).

```python
"sizes": [("original", None), ("100k", 100_000), ("10k", 10_000)]
```

This is how you study **data efficiency** — train the same model on progressively smaller
subsets and compare the curve.

## Subword models and the `+bytes` shorthand

`subword_models` accepts: `word`, `char`, `bytes`, `bpe`, `unigram`, and the byte-fallback
variants `char+bytes`, `unigram+bytes`. The `+bytes` suffix is sugar for setting
`byte_fallback=True` on that entry — equivalent to:

```python
{"subword_models": ["unigram"], "byte_fallback": True}
```

Byte fallback emits unseen characters as raw bytes instead of `<unk>`, which matters for
rare scripts, emoji, and names. The flag is orthogonal to the model, so to compare BPE
*with and without* fallback you declare two entries:

```python
encoding=[
    {"subword_models": ["bpe"], "vocab_sizes": [8000]},                      # bpe/8000
    {"subword_models": ["bpe"], "vocab_sizes": [8000], "byte_fallback": True},# bpe+bytes/8000
]
```

The two land at different on-disk paths (`bpe/8000` vs `bpe+bytes/8000`), so they're fully
independent cells.

!!! note "`word` triggers Moses pretokenization"
    The `word` model whitespace-tokenizes with Moses before building the vocabulary. It
    produces a large vocab but a fast model. `bytes` ignores `vocab_sizes` entirely (the
    vocab is always 256).

## Iterating over cells

The experiment loop is the same regardless of grid size — iterate the train datasets,
build vocabs, train, predict, collect scores:

```python
scores = []
for train_ds in builder.get_train_ds():
    src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)
    trainer = AutonmtTranslator.from_dataset(
        train_ds, model=Transformer.from_vocabs(src_vocab, tgt_vocab),
        src_vocab=src_vocab, tgt_vocab=tgt_vocab, run_prefix="grid",
    )
    trainer.fit(train_ds, config=fit_cfg)
    scores.append(trainer.predict(builder.get_test_ds(), config=pred_cfg))

generate_report(scores=scores, output_path="outputs/grid")
```

## Choosing what to evaluate: `eval_mode`

`PredictConfig(eval_mode=...)` controls which test sets each trained model is scored on:

| `eval_mode` | Scores on |
| --- | --- |
| `"same"` | Only the test set with the same `(name, lang, size)` as the trained cell |
| `"compatible"` | Every test set with the **same language pair** |
| `"all"` | Every test set |

`"compatible"` is the most useful for cross-corpus generalization: train on small subsets,
evaluate on the full test sets of every compatible corpus you have. With the 6-cell grid
above and two size variants, `"compatible"` produces `6 × 2 = 12` report rows.

→ Full runnable example: [`examples/basics/05_full_grid.py`](https://github.com/salvacarrion/autonmt/blob/main/examples/basics/05_full_grid.py)
and the [Grid experiments guide](../guides/grid-experiments.md).
