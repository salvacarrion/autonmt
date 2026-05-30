# Installation

AutoNMT requires **Python 3.8+**. CI tests against 3.9, 3.11 and 3.12; other 3.x versions
should work but are not actively verified.

## Pip (editable)

The recommended way to install AutoNMT for research is an editable checkout, so you can
read and tweak the source:

```bash
git clone https://github.com/salvacarrion/autonmt.git
cd autonmt/
pip install -e .
```

This pulls in the runtime dependencies only - PyTorch, PyTorch Lightning, SentencePiece,
sacreBLEU, Moses (`sacremoses`), BERTScore, and the plotting/IO stack.

## Optional dependencies

AutoNMT keeps the base install lean. Anything used by a single feature lives behind an
[extra](https://peps.python.org/pep-0508/#extras) you opt into:

| Extra        | Install                         | Unlocks                                                                     |
| ------------ | ------------------------------- | --------------------------------------------------------------------------- |
| `hf`         | `pip install -e '.[hf]'`        | `download_hf_dataset()` - pull a HuggingFace corpus into the AutoNMT layout |
| `hf-models`  | `pip install -e '.[hf-models]'` | `HuggingFaceTranslator` - evaluate / fine-tune pretrained seq2seq models    |
| `wandb`      | `pip install -e '.[wandb]'`     | Weights & Biases training logger (`wandb_params=` on `fit()`)               |
| `fairseq` ⚠️ | `pip install -e '.[fairseq]'`   | `FairseqTranslator` - **deprecated**, kept for backwards compatibility      |
| `docs`       | `pip install -e '.[docs]'`      | Build this documentation site locally                                       |
| `dev`        | `pip install -e '.[dev]'`       | `pytest` + `flake8` for development                                         |
| `all`        | `pip install -e '.[all]'`       | Every runtime extra above in one shot                                       |

!!! warning "Fairseq is deprecated"
Fairseq was [archived by its maintainers on 2026-03-20](https://github.com/facebookresearch/fairseq).
`FairseqTranslator` still works for existing flows, but new projects should use the
AutoNMT Lightning backend. See [Fairseq backend](../backends/fairseq.md).

## Docker

A `Dockerfile` is provided for a reproducible, GPU-ready environment:

```bash
git clone https://github.com/salvacarrion/autonmt.git
cd autonmt/
docker build -t autonmt:latest .
docker run --gpus all -d -v "$PWD":/autonmt --name autonmt_container autonmt:latest
docker exec -it autonmt_container bash
```

## Verify the install

```bash
python -c "import autonmt; print(autonmt.__version__)"
pytest tests/        # optional: run the test suite (needs the [dev] extra)
```

The synthetic-corpus end-to-end test exercises the full preprocessing pipeline in under
two seconds and needs no GPU - a quick way to confirm the install is healthy:

```bash
pytest tests/functional/test_builder_e2e.py -v
```

Next: the [Quickstart](quickstart.md).
