<div align="center">

<img src="https://github.com/salvacarrion/autonmt/raw/main/docs/images/logos/logo_with_name.png" alt="AutoNMT" width="280"/>

**A framework to streamline the research of neural machine translation models.**

[![Build](https://github.com/salvacarrion/autonmt/actions/workflows/python-package.yml/badge.svg)](https://github.com/salvacarrion/autonmt/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/github/license/salvacarrion/autonmt)](LICENSE)
[![Release](https://img.shields.io/github/v/release/salvacarrion/autonmt)](https://github.com/salvacarrion/autonmt/releases)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://salvacarrion.github.io/autonmt/)

[Documentation](https://salvacarrion.github.io/autonmt/) ·
[Quickstart](#quickstart) ·
[Examples](examples) ·
[Report a bug](https://github.com/salvacarrion/autonmt/issues)

</div>

---

AutoNMT is a modular research toolkit that takes the repetitive half of NMT experimentation - tokenization, training, scoring, logging, plotting, file management - off your hands so you can focus on the model. Declare a grid of datasets × language pairs × subword models × vocab sizes, and AutoNMT runs the cross-product, persists every intermediate artifact on disk, and produces a single comparable report at the end.

Every layer - datasets, vocabularies, models, decoding, metrics, reports - is designed to be subclassed, replaced, or extended via callable hooks, so researchers can plug in custom components without forking the core. The same script can train AutoNMT's own PyTorch Lightning models, fine-tune HuggingFace seq2seq checkpoints, or shell out to Fairseq - backends are swapped by changing one class.

## Highlights

- **Grid-first API** - describe an experiment as a cross-product, not a for-loop.
- **Pluggable backends** - `AutonmtTranslator` (Lightning), `HuggingFaceTranslator`, `FairseqTranslator` (deprecated).
- **Reproducible by construction** - every stage writes to a numbered folder; re-runs skip completed steps.
- **Subword variants out of the box** - `word`, `char`, `bytes`, `bpe`, `unigram`, with optional byte fallback.
- **Built-in evaluation** - sacreBLEU, BERTScore, COMET, HuggingFace metrics, wired into the report.
- **Extension-friendly core** - subclass `LitSeq2Seq` or pass callable hooks instead of patching internals.

## Installation

Requires Python 3.12+.

```bash
pip install -e .                       # core
pip install -e '.[hf]'                 # HuggingFace dataset loader
pip install -e '.[hf-models]'          # HuggingFace translator backend
pip install -e '.[wandb]'              # W&B logger
pip install -e '.[all]'                # everything above
```

See the [installation guide](https://salvacarrion.github.io/autonmt/get-started/installation/) for optional extras and GPU notes.

## Quickstart

Fetch a dataset from HuggingFace, train a small Transformer, and score it - in one script:

```python
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.models import Transformer

download_hf_dataset(
    hf_id="bentrevett/multi30k", base_path="datasets/quickstart",
    dataset_name="multi30k", lang_pair="de-en", src_field="de", tgt_field="en",
)

builder = DatasetBuilder(
    base_path="datasets/quickstart",
    datasets=[{"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]}],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [4000]}],
).build()

train_ds = builder.get_train_ds()[0]
src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)

trainer = AutonmtTranslator.from_dataset(
    train_ds,
    model=Transformer.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    run_prefix="quickstart",
)
trainer.fit(train_ds, config=FitConfig(max_epochs=3, batch_size=128))
scores = trainer.predict(builder.get_test_ds(), config=PredictConfig(metrics={"bleu"}))
```

Full walkthroughs live in [`examples/`](examples) - a step-by-step tutorial that builds from this snippet up to a multi-axis grid and a HuggingFace backend swap.

## Documentation

Full docs are published at **[salvacarrion.github.io/autonmt](https://salvacarrion.github.io/autonmt/)**:

- [Get started](https://salvacarrion.github.io/autonmt/get-started/installation/) - install, first experiment, understanding the output.
- [User guide](https://salvacarrion.github.io/autonmt/guide/experiments/workflow/) - data, models, training, translation, evaluation, backends.
- [How-to guides](https://salvacarrion.github.io/autonmt/how-to/) - task-oriented recipes for common workflows.
- [Concepts](https://salvacarrion.github.io/autonmt/concepts/philosophy/) - design philosophy, mental model, architecture, on-disk layout, reproducibility.
- [API reference](https://salvacarrion.github.io/autonmt/reference/) - autodoc from docstrings.

## Contributing

Contributions are welcome - bug reports, feature requests, docs fixes, or new backends/metrics. See [`docs/contributing.md`](docs/contributing.md) for the dev setup, test commands, and PR conventions.

```bash
pytest tests/
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Citation

If you use AutoNMT in academic work, please cite:

```bibtex
@misc{carrión2023autonmtframeworkstreamlineresearch,
      title={AutoNMT: A Framework to Streamline the Research of Seq2Seq Models},
      author={Salvador Carrión and Francisco Casacuberta},
      year={2023},
      eprint={2302.04981},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2302.04981},
}
```

## License

[MIT](LICENSE) © Salva Carrión.
