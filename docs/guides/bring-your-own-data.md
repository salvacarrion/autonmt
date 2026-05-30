# Bring your own data

The quickstart downloaded multi30k from HuggingFace, but most real projects start from
files you already have. AutoNMT doesn't care where the corpus came from - it only cares
that the files sit in the right place in the [on-disk layout](../concepts/on-disk-layout.md).

Mirrors [`examples/basics/02_bring_your_own_data.py`](https://github.com/salvacarrion/autonmt/blob/main/examples/basics/02_bring_your_own_data.py).

## Where your files go

Drop your splits into `1_splits/` under the cell's path, using the **language code as the
file extension**:

```text
datasets/mycorpus/es-en/original/
└── data/
    └── 1_splits/
        ├── train.es   train.en
        ├── val.es     val.en
        └── test.es    test.en
```

That's it. There is no manifest to write - the path _is_ the metadata. The segments encode
the dataset name (`mycorpus`), language pair (`es-en`), and size variant (`original`).

!!! tip "Only have raw text?"
If you have a single unsplit corpus instead of `train/val/test`, put it in `0_raw/`
(`data.es`, `data.en`) and let the builder create the splits. Provide a
`preprocess_raw_fn` if you want to shuffle/dedupe before splitting.

## Point the builder at it

The builder declaration is identical to the quickstart - just skip the download:

```python
from autonmt.datasets import DatasetBuilder

builder = DatasetBuilder(
    base_path="datasets/mycorpus",
    datasets=[{"name": "mycorpus", "languages": ["es-en"], "sizes": [("original", None)]}],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [8000]}],
    preprocess_raw_fn=preprocess_train,
    preprocess_splits_fn=preprocess_train,
    merge_vocabs=False,
).build(force_overwrite=False)
```

`build()` picks up your `1_splits/` files, normalizes them into `2_preprocessed/`, trains
SentencePiece, and writes `4_encoded/`. From here the rest of the pipeline (train,
predict, report) is exactly as in the [Quickstart](../getting-started/quickstart.md).

## Multiple corpora and language pairs

Add more entries to `datasets` (or more pairs to `languages`) and they all unroll into the
grid. You can register a held-out test-only corpus the same way and reach it with
`eval_mode="compatible"` - see [The grid](../concepts/grid.md#choosing-what-to-evaluate-eval_mode).

```python
datasets=[
    {"name": "europarl",  "languages": ["es-en"], "sizes": [("original", None)]},
    {"name": "newstest",  "languages": ["es-en"], "sizes": [("original", None)]},  # eval-only corpus
]
```

## Guard against train/test leakage

If you assembled splits yourself, it's easy to leak test pairs into training. AutoNMT ships
a checker:

```python
from autonmt.datasets.leakage import warn_on_leakage

warn_on_leakage(train_ds)   # logs a warning if test lines appear in train
```

→ See [`autonmt.datasets.leakage`](../reference/datasets.md) for `find_leaked_lines` (the
function that returns the offending lines rather than just warning).

## Loading from HuggingFace

When you _do_ want a Hub corpus, `download_hf_dataset` writes it straight into the layout
(needs the `hf` extra):

```python
from autonmt.datasets.hf_loader import download_hf_dataset

download_hf_dataset(
    hf_id="bentrevett/multi30k", base_path="datasets/multi30k",
    dataset_name="multi30k", lang_pair="de-en",
    src_field="de", tgt_field="en",
)
```
