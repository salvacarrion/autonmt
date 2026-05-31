# Use a custom dataset

The [`DatasetBuilder`](../guide/data/datasets.md) reads parallel text from disk; how the
files *got* there is up to you. The only contract is the **layout** and the **language-code
extension**.

## From your own pre-split files

Place `train` / `val` / `test` in the splits folder, using the language code as the file
extension, then point the builder at the parent:

```text
data/mycorpus/es-en/original/data/1_splits/
├── train.es   train.en
├── val.es     val.en
└── test.es    test.en
```

```python
from autonmt.datasets import DatasetBuilder

builder = DatasetBuilder(
    base_path="data",
    datasets=[{"name": "mycorpus", "languages": ["es-en"], "sizes": [("original", None)]}],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [8000]}],
).build()
```

`.build()` picks up your splits and runs preprocessing → encoding → vocab on top of them.

## From a single unsplit file pair

If you only have one parallel pair (not yet split into train/val/test), drop it in `0_raw/`
instead and the builder creates the splits for you:

```text
data/mycorpus/es-en/original/data/0_raw/
├── data.es
└── data.en
```

## From the HuggingFace Hub

With the `[hf]` extra, `download_hf_dataset` writes a Hub corpus straight into the layout:

```python
from autonmt.datasets.hf_loader import download_hf_dataset

download_hf_dataset(
    hf_id="bentrevett/multi30k", base_path="data",
    dataset_name="multi30k", lang_pair="de-en",
    src_field="de", tgt_field="en",
    # split_map={"train": "train", "val": "validation", "test": "test"},  # if names differ
    # max_train_lines=50000,                                              # truncate for smoke tests
)
```

It handles flat columns and the nested `{"translation": {"de": …, "en": …}}` convention
(pass the leaf key as `src_field`/`tgt_field`).

## Clean it your way

Inject custom normalization/filtering with the builder's [hooks](../guide/data/preprocessing.md#hooks)
— no subclassing:

```python
builder = DatasetBuilder(
    base_path="data", datasets=[...], encoding=[...],
    preprocess_raw_fn=clean_pairs,       # fn(data, ds) -> (src_lines, tgt_lines)
    preprocess_splits_fn=clean_pairs,
)
```

!!! tip "Check for leakage first"
    Overlap between train and test silently inflates scores. Run the cheap checker before
    spending GPU hours:
    ```python
    from autonmt.datasets.leakage import warn_on_leakage
    warn_on_leakage(train_lines, test_lines, key_fn=str.lower, label="es-en tgt")
    ```

Full details: [Datasets & the dataset builder](../guide/data/datasets.md#bring-your-own-data).
