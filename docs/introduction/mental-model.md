# The mental model

If you remember one diagram from these docs, make it this one.

```mermaid
flowchart LR
    A["<b>Grid declaration</b><br/>datasets × pairs<br/>× subword × vocab"] --> B["<b>DatasetBuilder</b><br/>unroll + prepare<br/>on disk"]
    B --> C["<b>Dataset variants</b><br/>one per cell"]
    C --> D["<b>Translator</b><br/>fit · predict<br/>(backend of choice)"]
    D --> E["<b>Scores</b><br/>per run × eval set"]
    E --> F["<b>Report</b><br/>json · csv · plots"]
```

Everything in AutoNMT is a station on this line. Reading it left to right *is* the mental
model: **a grid becomes dataset variants, variants flow through a translator, and the
translator's scores become a report.**

## The three layers

### 1. The grid → dataset variants

You declare the **axes** of your experiment. The
[`DatasetBuilder`](../data/dataset-builder.md) computes the cross-product and turns each
cell into a [`Dataset`](../data/dataset-builder.md#the-dataset-object) object. A `Dataset`
is not a PyTorch dataset — it's an **identity + a path engine**. Given *(name, language
pair, size, subword model, vocab size)* it knows where every file for that cell lives on
disk, and the builder materializes those files (clean → split → encode → build vocab).

```python
builder = DatasetBuilder(base_path="data", datasets=[...], encoding=[...]).build()
train_variants = builder.get_train_ds()   # list of Dataset, one per cell
test_variants  = builder.get_test_ds()
```

### 2. Dataset variants → translator

A [translator](../backends/index.md) is the thing that turns a dataset variant into a
trained model and then into translations. It exposes exactly two verbs:

- **`fit(train_ds, ...)`** — train (or fine-tune) on a variant.
- **`predict(eval_datasets, ...)`** — translate the test set(s) and score the output.

Which translator you instantiate decides *which NMT toolkit* runs underneath — AutoNMT's
own Lightning engine, HuggingFace, or Fairseq — but the two verbs never change. This is
the [Keras-style abstraction](philosophy.md#keras): the loop you write is backend-agnostic.

### 3. Translator → report

`predict()` returns a list of score dicts (one per run × evaluation set). Feed that list
to [`generate_report`](../evaluation/reports.md) and you get JSON + CSV summaries and
comparison plots — every cell scored identically, side by side.

## A whole experiment is a flat loop

Because the builder already unrolled the grid, your experiment is **iteration, not
nesting**:

```python
from autonmt.reporting.report import generate_report

scores = []
for train_ds in builder.get_train_ds():          # one cell of the grid
    src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=8000)
    trainer = AutonmtTranslator.from_dataset(
        train_ds,
        model=Transformer.from_vocabs(src_vocab, tgt_vocab),
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        run_prefix="sweep",
    )
    trainer.fit(train_ds, config=FitConfig(max_epochs=10))
    scores.append(trainer.predict(builder.get_test_ds(), config=PredictConfig(metrics={"bleu", "chrf"})))

generate_report(scores=scores, output_path="outputs/sweep", plot_metric="translations.beam5.sacrebleu_bleu_score")
```

No matter how many axes you added to the grid, the loop body is the same. That's the whole
point: **the complexity lives in the declaration, not in your control flow.**

## Why the docs are organized the way they are

The pipeline is also the table of contents. Reading the diagram left to right:

- **Data goes in** → [Data & vocabularies](../data/dataset-builder.md) covers the input
  side: the builder, preprocessing/encoding (subwords), and vocabularies.
- **The middle swaps** → [The AutoNMT toolkit](../toolkit/overview.md) documents the
  native engine in depth, and [Backends](../backends/index.md) covers choosing/configuring
  HuggingFace or Fairseq instead. Only this stage changes between backends.
- **Reports come out** → [Evaluation & reports](../evaluation/metrics.md) covers the output
  side: metrics, significance testing, and the report itself.

Wrapping all of it: [Architecture](../architecture/building-blocks.md) shows how the pieces
compose and how a backend plugs in, and explains the on-disk layout that makes the whole
thing reproducible.

!!! tip "Where do datasets / vocab / metrics / reporting live in these docs?"
    They're split along the pipeline rather than grouped into one section. Datasets and
    vocabularies sit **before** the engine (they're the input you need first); metrics and
    reports sit **after** it (they're the output). That ordering mirrors the diagram — and
    reinforces the key idea that input and output are shared while only the toolkit in the
    middle is interchangeable.

---

Ready to run it? Head to **[Get started → Installation](../get-started/installation.md)**.
Want the structural view first? Go to **[Architecture](../architecture/building-blocks.md)**.
