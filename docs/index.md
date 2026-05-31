<p align="center">
  <img src="images/logos/logo_with_name.png" alt="AutoNMT" width="400">
</p>

<p style="font-size: 1.15rem; opacity: 0.85; text-align: center;">
A framework to streamline the research of neural machine translation models.
</p>

AutoNMT takes the **repetitive half** of NMT research off your hands — tokenization,
training, decoding, scoring, logging, plotting, and the file bookkeeping that ties them
together — so the hours you spend are the hours that actually move your research: the
model.

You don't write loops over datasets and hyper-parameters. You **declare a grid** —
datasets × language pairs × subword models × vocabulary sizes — and AutoNMT walks the
cross-product, persists every intermediate artifact in a predictable place on disk, and
hands you one comparable report at the end. Swap a single class and the same script
trains AutoNMT's own PyTorch Lightning models, fine-tunes a HuggingFace checkpoint, or
shells out to Fairseq.

```python
from autonmt.datasets import DatasetBuilder
from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.nn.models import Transformer

# 1. Declare the grid → AutoNMT unrolls it into dataset variants on disk.
builder = DatasetBuilder(
    base_path="data",
    datasets=[{"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]}],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [4000]}],
).build()

train_ds = builder.get_train_ds()[0]
src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)

# 2. Bind a model to a backend and run the experiment loop.
trainer = AutonmtTranslator.from_dataset(
    train_ds,
    model=Transformer.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    run_prefix="demo",
)
trainer.fit(train_ds, config=FitConfig(max_epochs=3, batch_size=128))
scores = trainer.predict(builder.get_test_ds(), config=PredictConfig(metrics={"bleu"}))
```

That snippet is the whole shape of an AutoNMT experiment: **describe the data, pick a
backend, `fit`, `predict`.** Everything in these docs is about understanding what happens
inside those four steps and how to bend each one to your research.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Get started**

    ---

    Install AutoNMT and run a complete experiment — train, translate, score, report — in
    a few minutes.

    [:octicons-arrow-right-24: Installation & quickstart](get-started/installation.md)

-   :material-book-open-page-variant:{ .lg .middle } **User guide**

    ---

    The pipeline, block by block: data, models, training, translation, evaluation — and
    the backend you run them on.

    [:octicons-arrow-right-24: The experiment workflow](guide/experiments/workflow.md)

-   :material-lightbulb-on:{ .lg .middle } **Concepts**

    ---

    The mental model, the design philosophy, the toolkit abstraction, and the on-disk
    layout behind reproducibility.

    [:octicons-arrow-right-24: How AutoNMT thinks](concepts/mental-model.md)

-   :material-swap-horizontal:{ .lg .middle } **Backends**

    ---

    When to use the native engine, when to fine-tune HuggingFace, and the deprecated
    Fairseq path — all behind one `fit` / `predict`.

    [:octicons-arrow-right-24: Choosing a backend](guide/backends/choosing.md)

-   :material-chart-bar:{ .lg .middle } **Evaluation & reports**

    ---

    Metrics (BLEU, chrF, TER, COMET, BERTScore), significance testing, and comparable
    reports with plots.

    [:octicons-arrow-right-24: Metrics](guide/evaluation/metrics.md)

-   :material-api:{ .lg .middle } **API reference**

    ---

    Signatures, parameters, and public contracts for every module, generated from the
    source docstrings.

    [:octicons-arrow-right-24: Reference](reference/index.md)

</div>

---

!!! note "Who this is for"
    AutoNMT is built for **researchers**. We assume you can write Python and train a
    model, but we do *not* assume every reader is equally fluent in every NMT concept.
    Whenever a piece of machine-translation machinery shows up — subwords, beam search,
    samplers, length penalties — you'll find a short, intuitive primer right next to it,
    so you can keep reading without a detour to a textbook.

    **New to neural machine translation?** Start with
    [How NMT works](get-started/how-nmt-works.md) — a 10-minute, first-principles tour of
    the whole pipeline that every other page links back to.

!!! quote "Citing AutoNMT"
    If you use AutoNMT in academic work, please cite
    [Carrión & Casacuberta (2023)](https://arxiv.org/abs/2302.04981):

    ```bibtex
    @misc{carrion2023autonmt,
      title  = {AutoNMT: A Framework to Streamline the Research of Seq2Seq Models},
      author = {Salvador Carrión and Francisco Casacuberta},
      year   = {2023},
      eprint = {2302.04981},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CL},
      url = {https://arxiv.org/abs/2302.04981}
    }
    ```
