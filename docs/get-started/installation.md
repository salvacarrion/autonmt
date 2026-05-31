# Installation

AutoNMT requires **Python 3.12+**.

## Core install

From a clone of the repository (editable install — recommended while the API is still
moving):

```bash
git clone https://github.com/salvacarrion/autonmt.git
cd autonmt
pip install -e .
```

The core pulls in everything you need to prepare data, train AutoNMT's own models, decode,
and score with sacreBLEU / chrF / TER / BERTScore: `torch`, `pytorch-lightning`,
`sentencepiece`, `sacremoses`, `sacrebleu`, `bert-score`, plus the plotting stack
(`matplotlib`, `seaborn`) and `tensorboard`.

!!! tip "Use a fresh environment"
    A virtual environment (or conda env) per project keeps PyTorch and CUDA versions from
    colliding across experiments.
    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install -e .
    ```

## Optional extras

AutoNMT keeps the base install lean. Features that need heavier dependencies are opt-in
extras you add in brackets:

```bash
pip install -e '.[hf]'           # download HF datasets + HuggingFace metrics (hg_*)
pip install -e '.[hf-models]'    # the HuggingFaceTranslator backend (fine-tune / evaluate)
pip install -e '.[wandb]'        # Weights & Biases logger
pip install -e '.[docs]'         # build this documentation site locally
pip install -e '.[all]'          # hf + hf-models + wandb + dev tooling
```

| Extra        | Unlocks                                                                 |
| ------------ | ---------------------------------------------------------------------- |
| `hf`         | `download_hf_dataset(...)` (pull a corpus from the Hub) and `hg_*` metrics |
| `hf-models`  | [`HuggingFaceTranslator`](../guide/backends/huggingface.md) — load/fine-tune `AutoModelForSeq2SeqLM` |
| `wandb`      | W&B logging via `fit(..., wandb_params=...)`                            |
| `docs`       | MkDocs Material + `mkdocstrings` for `mkdocs serve` / `build`           |
| `dev`        | `pytest`, `flake8`                                                      |

### Dependencies that aren't extras

A couple of optional capabilities pull large models you'll usually want to install on
their own:

```bash
pip install unbabel-comet      # COMET metric (downloads a ~2 GB model on first use)
```

[Fairseq](../guide/backends/fairseq.md) is intentionally **not** part of the default
install and not in the `all` extra — it was archived upstream and is deprecated. If you
need it for an existing flow:

```bash
pip install -e '.[fairseq]'    # deprecated; prefer AutonmtTranslator for new work
```

## Verify the install

```bash
python -c "import autonmt; print(autonmt.__version__)"
```

```bash
pytest tests/      # the test suite is hermetic and fast — no GPU, no downloads
```

The functional suite includes a synthetic end-to-end run of the preprocessing pipeline, so
a green `pytest` confirms the data path works on your machine.

## GPU notes

AutoNMT follows whatever PyTorch you install. The native engine and the HuggingFace backend
both resolve the device automatically (`accelerator="auto"` picks CUDA → MPS → CPU), so the
same script runs on a GPU box, an Apple-silicon laptop, or CPU-only CI without changes. To
pin a device explicitly, pass `accelerator=` / `devices=` through
[`FitConfig`](../guide/training/training.md) / [`PredictConfig`](../guide/translation/generating.md).

---

Installed? Run your **[first experiment](quickstart.md)**.
