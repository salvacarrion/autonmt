# Contributing

Contributions are welcome — bug reports, feature requests, docs fixes, or new
backends/metrics.

## Development setup

```bash
git clone https://github.com/salvacarrion/autonmt.git
cd autonmt
pip install -e '.[dev]'
```

## Tests

```bash
pytest tests/
```

The unit tests cover enums, config merging, dataset paths, vocabularies, search/Transformer
parity, and the HuggingFace mapping. The functional suite includes a **hermetic
synthetic-corpus end-to-end test** of the preprocessing pipeline (raw → splits →
SentencePiece → encoded) that runs in a couple of seconds with no GPU — handy for verifying
refactors quickly:

```bash
pytest tests/functional/test_builder_e2e.py -v
```

## Lint

CI only fails the build on syntax errors and undefined names:

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Building the docs locally

```bash
pip install -e '.[docs]'
mkdocs serve            # live-reload at http://127.0.0.1:8000
mkdocs build --strict   # catch broken internal links before pushing
```

The API reference is generated from docstrings with
[mkdocstrings](https://mkdocstrings.github.io/) using **NumPy-style** docstrings — a new
public class or function is documented for free, just write a clear docstring with
`Parameters` / `Returns` sections.

## Project conventions

A few conventions worth knowing (see also `CLAUDE.md` in the repo root):

- **`assert` statements stay.** They catch shape/identity bugs during research iteration;
  don't strip them.
- **Examples are self-contained.** Each `examples/*.py` duplicates its boilerplate on purpose
  — don't extract shared helpers.
- **Keep the core minimal.** Prefer [extension points](extending/index.md) (callable hooks,
  subclassing) over new built-in flags.
- **Language code as the file extension.** Dataset files are `train.es` / `train.en`, never
  `es/train.txt`.
- **Config dumps keep primitives only.** `logs/config_{train,predict}.json` renders callables
  as `module.qualname` but won't round-trip arbitrary objects.
- **Don't add Fairseq to the default install.** It's deprecated and lives behind the
  `[fairseq]` extra.

## License

[MIT](https://github.com/salvacarrion/autonmt/blob/main/LICENSE) © Salva Carrión.
