# Your first experiment

This page walks through a complete, minimal AutoNMT run — **fetch a corpus, train a small
Transformer, score it, and write a report** — explaining every block as we go. It needs the
HuggingFace dataset loader, so install the `hf` extra first:

```bash
pip install -e '.[hf]'
```

We'll translate **German → English** on [Multi30k](https://github.com/multi30k/dataset), a
small, well-behaved dataset that trains in minutes on CPU.

## The whole script

Here it is end to end; the sections below unpack each part.

```python
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.nn.models import Transformer
from autonmt.reporting.report import generate_report, format_summary_table

# 1. Get a parallel corpus onto disk in AutoNMT's layout.
download_hf_dataset(
    hf_id="bentrevett/multi30k", base_path="data",
    dataset_name="multi30k", lang_pair="de-en",
    src_field="de", tgt_field="en",
)

# 2. Declare the (tiny) grid and materialize it.
builder = DatasetBuilder(
    base_path="data",
    datasets=[{"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]}],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [4000]}],
).build()

# 3. Pick the single variant and build its vocabularies.
train_ds = builder.get_train_ds()[0]
src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)

# 4. Bind a model to the AutoNMT backend.
trainer = AutonmtTranslator.from_dataset(
    train_ds,
    model=Transformer.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    run_prefix="first",
)

# 5. Train.
trainer.fit(train_ds, config=FitConfig(max_epochs=5, batch_size=128, learning_rate=1e-3))

# 6. Translate + score the test set.
scores = trainer.predict(
    builder.get_test_ds(),
    config=PredictConfig(beams=[5], metrics={"bleu", "chrf"}),
)

# 7. Write the report and print the summary.
df_report, df_summary = generate_report(scores=[scores], output_path="outputs/first")
print(format_summary_table(df_summary))
```

## Step by step

### 1 · Get the data onto disk

```python
download_hf_dataset(
    hf_id="bentrevett/multi30k", base_path="data",
    dataset_name="multi30k", lang_pair="de-en",
    src_field="de", tgt_field="en",
)
```

[`download_hf_dataset`](../data/dataset-builder.md#bring-your-own-data) pulls the corpus
from the Hub and writes it as `train` / `val` / `test` splits in the exact directory layout
the builder expects:

```
data/multi30k/de-en/original/data/1_splits/{train,val,test}.{de,en}
```

The filename **extension is the language code** (`train.de`, `train.en`) — that's an
AutoNMT convention you'll see everywhere. If your data is already on disk in that layout,
skip this step entirely (see [Bring your own
data](../data/dataset-builder.md#bring-your-own-data)).

### 2 · Declare and build the grid

```python
builder = DatasetBuilder(
    base_path="data",
    datasets=[{"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]}],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [4000]}],
).build()
```

This grid has **one** cell, but it's declared the same way a 16-cell sweep would be — that's
the [grid-first idea](../introduction/philosophy.md#grid-first). `.build()` runs the data
pipeline for the cell: clean → split → learn a BPE tokenizer → encode every split → build
the vocab artifacts, all written to numbered folders under `data/multi30k/de-en/original/`.

!!! info "What's a subword model? (BPE in one paragraph)"
    A model can't have an entry for every word — vocabularies would be huge and every rare
    or unseen word would become `<unk>`. **Subword tokenization** splits text into reusable
    fragments instead. **BPE** (Byte-Pair Encoding) starts from characters and repeatedly
    merges the most frequent adjacent pair, learning a vocabulary of common pieces:
    `"lower"` might become `lo + wer`, and an unseen `"lowest"` still encodes as `lo + west`
    — no `<unk>`. `vocab_sizes=[4000]` asks for 4 000 such pieces. AutoNMT also supports
    `word`, `char`, `bytes`, `unigram`, and byte-fallback variants — see
    [Preprocessing & subword encoding](../data/preprocessing-and-encoding.md).

### 3 · Pick the variant, build vocabularies

```python
train_ds = builder.get_train_ds()[0]
src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)
```

`get_train_ds()` returns the list of dataset variants (here, length 1). Each
[`Dataset`](../data/dataset-builder.md#the-dataset-object) knows where its encoded files and
vocab artifacts live. `build_vocabs` loads the source/target
[vocabularies](../data/vocabularies.md); `max_tokens` caps how many tokens a single sentence
contributes when encoding batches later.

### 4 · Bind a model to a backend

```python
trainer = AutonmtTranslator.from_dataset(
    train_ds,
    model=Transformer.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    run_prefix="first",
)
```

[`AutonmtTranslator`](../toolkit/overview.md) is the native (PyTorch Lightning) backend.
`from_dataset(...)` wires the translator to the variant's on-disk run location (so
checkpoints, logs, and translations land in the right place) and tags this run `first_*`.
[`Transformer.from_vocabs`](../toolkit/models.md) builds a small Transformer sized to the
vocabularies. To run on HuggingFace or Fairseq instead, you'd swap **only this object** —
see [Backends](../backends/index.md).

### 5 · Train

```python
trainer.fit(train_ds, config=FitConfig(max_epochs=5, batch_size=128, learning_rate=1e-3))
```

[`fit`](../toolkit/training.md) trains the model, handling DataLoaders, the optimizer,
checkpointing (best/last), and logging for you. Everything tunable lives in
[`FitConfig`](../toolkit/training.md#fitconfig); here we set just three fields and take
defaults for the rest. The best checkpoint is saved under the run's `checkpoints/` folder.

### 6 · Translate and score

```python
scores = trainer.predict(
    builder.get_test_ds(),
    config=PredictConfig(beams=[5], metrics={"bleu", "chrf"}),
)
```

[`predict`](../toolkit/predict.md) decodes the test set with **beam search** (width 5) and
scores the output with **BLEU** and **chrF**. It returns a list of score dicts — one per
evaluation set.

!!! info "What's beam search? (and BLEU/chrF)"
    Generating a translation greedily — always taking the single most likely next token —
    can paint the model into a corner. **Beam search** instead keeps the *k* best partial
    translations ("beams") at every step and expands them all, then returns the
    highest-scoring complete one. **BLEU** measures n-gram overlap with the reference
    translation (higher is better); **chrF** does the same at the character level and is
    kinder to morphologically rich languages. Both, and the math behind beam search, are
    covered in [Decoding strategies](../toolkit/decoding.md) and
    [Metrics](../evaluation/metrics.md).

### 7 · Report

```python
df_report, df_summary = generate_report(scores=[scores], output_path="outputs/first")
print(format_summary_table(df_summary))
```

[`generate_report`](../evaluation/reports.md) writes `report.json`, `report.csv`, and a
summary CSV under `outputs/first/reports/`, and `format_summary_table` prints a tidy table
to your terminal. With a grid bigger than one cell, this same call collects *every* run into
one comparable table — which is the entire reason AutoNMT exists.

## What landed on disk

After the run you'll find two trees:

```
data/multi30k/de-en/original/         # the dataset cell (reusable across runs)
├── data/{0_raw, 1_splits, 2_preprocessed, 4_encoded/bpe/4000/}
├── vocabs/bpe/4000/
└── models/autonmt/runs/first_bpe_4000/
    ├── checkpoints/                  # best/last .pt
    ├── logs/                         # TensorBoard + config_train.json / config_predict.json
    └── eval/.../translations/beam5/  # src.txt, ref.txt, hyp.txt, scores/
outputs/first/reports/                # report.json, report.csv, report_summary.csv
```

Notice the **separation**: the dataset cell is independent of the run, and each run is
self-describing (its config and environment are dumped next to its outputs). Re-run the
script and AutoNMT skips the stages already on disk. The layout is explained in full in
[On-disk layout & reproducibility](../architecture/layout-and-reproducibility.md).

---

That's the whole loop. To go deeper:

- **Understand the structure** → [Architecture](../architecture/building-blocks.md)
- **Scale the grid** → [Datasets & the dataset builder](../data/dataset-builder.md)
- **Tune training & decoding** → [The AutoNMT toolkit](../toolkit/overview.md)
