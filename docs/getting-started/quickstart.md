# Quickstart

This page takes you from a fresh install to a BLEU score in one script. It mirrors
[`examples/basics/01_hello_autonmt.py`](https://github.com/salvacarrion/autonmt/blob/main/examples/basics/01_hello_autonmt.py).

We'll fetch the **multi30k** German→English corpus from HuggingFace, train a small
Transformer for one epoch, and print a summary table.

!!! info "Install the HuggingFace loader first"
Downloading the corpus uses the `hf` extra:
`bash
    pip install -e '.[hf]'
    `

## The full script

```python
import datetime
import os

from tokenizers.normalizers import NFKC, Strip

from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.nn.models import Transformer
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.datasets.preprocessing import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.reporting.report import format_summary_table, generate_report

BASE_PATH = "datasets/quickstart"


# Three preprocessing callbacks AutoNMT calls at different pipeline stages.
def normalize(lines):
    return normalize_lines(lines, seq=[NFKC(), Strip()])

def preprocess_train(data, ds):
    return preprocess_pairs(data["src"]["lines"], data["tgt"]["lines"], normalize_fn=normalize)

def preprocess_predict(data, ds):
    return preprocess_lines(data["lines"], normalize_fn=normalize)


def main():
    # 1. Download multi30k into the AutoNMT on-disk layout.
    download_hf_dataset(
        hf_id="bentrevett/multi30k", base_path=BASE_PATH,
        dataset_name="multi30k", lang_pair="de-en",
        src_field="de", tgt_field="en",
    )

    # 2. Declare the (1-cell) grid: SentencePiece BPE-4000.
    builder = DatasetBuilder(
        base_path=BASE_PATH,
        datasets=[{"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]}],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [4000]}],
        preprocess_raw_fn=preprocess_train,
        preprocess_splits_fn=preprocess_train,
        merge_vocabs=False,
    ).build(force_overwrite=False)

    train_ds = builder.get_train_ds()[0]
    test_datasets = builder.get_test_ds()

    # 3. Train a tiny Transformer on the de→en cell.
    src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)
    model = Transformer.from_vocabs(src_vocab, tgt_vocab)

    trainer = AutonmtTranslator.from_dataset(
        train_ds, model=model,
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        run_prefix="hello",
    )
    trainer.fit(train_ds, config=FitConfig(max_epochs=1, batch_size=128, learning_rate=1e-3, seed=42))

    scores = trainer.predict(
        test_datasets,
        config=PredictConfig(
            metrics={"bleu"}, beams=[5],
            load_checkpoint="best",
            preprocess_fn=preprocess_predict,
            eval_mode="compatible",   # only score test sets with the same lang pair
        ),
    )

    # 4. Persist a report and print a one-line summary.
    out = f".outputs/quickstart/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    _, df_summary = generate_report(scores=[scores], output_path=out)
    print(f"\nReport saved to: {os.path.abspath(out)}\n")
    print(format_summary_table(df_summary))


if __name__ == "__main__":
    main()
```

Run it:

```bash
python examples/basics/01_hello_autonmt.py
```

!!! tip "A low score is expected here"
One epoch on a tiny Transformer will produce a **single-digit BLEU**. That's fine -
this script's job is to prove the pipeline runs end-to-end, not to produce a
publishable model.

## What just happened

Reading the script top to bottom is reading the [three-layer pipeline](../concepts/pipeline.md):

1. **`download_hf_dataset` + `DatasetBuilder`** prepared the corpus on disk: splits,
   normalization, a trained SentencePiece BPE-4000 tokenizer, and the encoded text. Every
   stage landed in a numbered folder under `datasets/quickstart/` - see the
   [on-disk layout](../concepts/on-disk-layout.md). Re-running skips finished stages.
2. **`AutonmtTranslator`** wrapped the `Transformer` and ran `fit()` (training) then
   `predict()` (translate the test set + score it).
3. **`generate_report`** flattened the per-run scores into JSON + CSV and a printable
   table.

## Next steps

- Don't want to download anything? [Bring your own data](../guides/bring-your-own-data.md).
- Want to control cleaning and tokenization? [Preprocessing & vocabularies](../guides/preprocessing-and-vocabs.md).
- Ready to compare configurations? [Grid experiments](../guides/grid-experiments.md).
