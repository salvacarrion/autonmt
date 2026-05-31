# Quickstart: your first experiment

This page runs a complete, minimal AutoNMT experiment end to end — **fetch a corpus, train
a small Transformer, score it, and write a report** — explaining each block as we go. It
uses the HuggingFace dataset loader, so install the `hf` extra first:

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
from autonmt.reporting.report import Report

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
report = Report.from_predict(scores, output_path="outputs/first").save()
print(report)
```

!!! info "Numbers will be low — that's expected"
    A small Transformer trained for a handful of epochs on a tiny corpus will score in the
    single digits. The point of this script is to prove the **pipeline runs end to end**,
    not to produce a publishable model.

## Step by step

### 1 · Get the data onto disk

```python
download_hf_dataset(
    hf_id="bentrevett/multi30k", base_path="data",
    dataset_name="multi30k", lang_pair="de-en",
    src_field="de", tgt_field="en",
)
```

[`download_hf_dataset`](../guide/data/datasets.md#bring-your-own-data) pulls the corpus
from the Hub and writes it as `train` / `val` / `test` splits in the exact directory layout
the builder expects:

```
data/multi30k/de-en/original/data/1_splits/{train,val,test}.{de,en}
```

The filename **extension is the language code** (`train.de`, `train.en`) — that's an
AutoNMT convention you'll see everywhere. If your data is already on disk in that layout,
skip this step entirely (see [Bring your own data](../guide/data/datasets.md#bring-your-own-data)).

### 2 · Declare and build the grid

```python
builder = DatasetBuilder(
    base_path="data",
    datasets=[{"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]}],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [4000]}],
).build()
```

This grid has **one** cell, but it's declared the same way a 16-cell sweep would be — that's
the [grid-first idea](../concepts/philosophy.md#grid-first). `.build()` runs the data
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
    [Subword tokenization](../guide/data/tokenization.md).

### 3 · Pick the variant, build vocabularies

```python
train_ds = builder.get_train_ds()[0]
src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)
```

`get_train_ds()` returns the list of dataset variants (here, length 1). Each
[`Dataset`](../guide/data/datasets.md#the-dataset-object) knows where its encoded files and
vocab artifacts live. `build_vocabs` loads the source/target
[vocabularies](../guide/data/vocabularies.md); `max_tokens` caps how many tokens a single
sentence contributes when encoding batches later.

### 4 · Bind a model to a backend

```python
trainer = AutonmtTranslator.from_dataset(
    train_ds,
    model=Transformer.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    run_prefix="first",
)
```

[`AutonmtTranslator`](../guide/backends/native.md) is the native (PyTorch Lightning)
backend. `from_dataset(...)` wires the translator to the variant's on-disk run location (so
checkpoints, logs, and translations land in the right place) and tags this run `first_*`.
[`Transformer.from_vocabs`](../guide/models/using-a-model.md) builds a small Transformer
sized to the vocabularies. To run on HuggingFace or Fairseq instead, you'd swap **only this
object** — see [Choosing a backend](../guide/backends/choosing.md).

### 5 · Train

```python
trainer.fit(train_ds, config=FitConfig(max_epochs=5, batch_size=128, learning_rate=1e-3))
```

[`fit`](../guide/training/training.md) trains the model, handling DataLoaders, the
optimizer, checkpointing (best/last), and logging for you. Everything tunable lives in
[`FitConfig`](../guide/training/training.md); here we set just three fields and take
defaults for the rest. The best checkpoint is saved under the run's `checkpoints/` folder.

### 6 · Translate and score

```python
scores = trainer.predict(
    builder.get_test_ds(),
    config=PredictConfig(beams=[5], metrics={"bleu", "chrf"}),
)
```

[`predict`](../guide/translation/generating.md) decodes the test set with **beam search**
(width 5) and scores the output with **BLEU** and **chrF**. It returns a list of score
dicts — one per evaluation set.

!!! info "What's beam search? (and BLEU/chrF)"
    Generating a translation greedily — always taking the single most likely next token —
    can paint the model into a corner. **Beam search** instead keeps the *k* best partial
    translations ("beams") at every step and expands them all, then returns the
    highest-scoring complete one. **BLEU** measures n-gram overlap with the reference
    translation (higher is better); **chrF** does the same at the character level and is
    kinder to morphologically rich languages. Both, and the math behind beam search, are
    covered in [Decoding strategies](../guide/translation/decoding.md) and
    [Metrics](../guide/evaluation/metrics.md).

### 7 · Report

```python
report = Report.from_predict(scores, output_path="outputs/first").save()
print(report)
```

[`Report`](../guide/evaluation/reports.md) writes `report.json`, `report.csv`, and
a summary CSV under `outputs/first/reports/`, and `print(report)` shows a tidy
table to your terminal. With a grid bigger than one cell, `Report.from_runs(...)` collects
*every* run into one comparable table — which is the entire reason AutoNMT exists.

---

That's the whole loop. Next, see exactly **[what landed on disk](understanding-the-output.md)**
and why it's organized that way — then scale the one-cell grid into a real sweep in the
[User guide](../guide/experiments/workflow.md).
