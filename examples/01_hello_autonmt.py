"""
============================================================================
 Tutorial 01 — Hello AutoNMT
============================================================================

What you'll learn
-----------------
The three blocks every AutoNMT experiment is made of:

    1. A `DatasetBuilder` that prepares files on disk (splits, vocab, encoded text).
    2. A `Translator` that wraps a model and runs `fit()` + `predict()`.
    3. A `generate_report` call that turns the scores into JSON/CSV + a table.

What's new vs the previous tutorial
-----------------------------------
This is the first one — no prior context assumed.

Run
---
    pip install -e '.[hf]'                  # for `datasets`
    python examples/01_hello_autonmt.py

Expected output
---------------
A few minutes of training logs, then a summary table printed to stdout and
a report written under `.outputs/01_hello/<timestamp>/`.

The BLEU score after a single epoch on a tiny Transformer is going to be
*low* (single digits). That's expected: this script's job is to prove the
pipeline runs end-to-end, not to produce a publishable model.
"""
import datetime
import os

from tokenizers.normalizers import NFKC, Strip

from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.nn.models import Transformer
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.datasets.processors import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.reporting.report import format_summary_table, generate_report

BASE_PATH = "datasets/01_hello"
DATASET = "multi30k"
LANG_PAIR = "de-en"


# AutoNMT calls these three callbacks at different stages of the pipeline:
#   - `normalize`  : char-level cleanup applied inside the other two.
#   - `preprocess_train` : runs on the train/val/test splits before SPM training.
#   - `preprocess_predict`: runs on the test source at predict time, before encoding.
def normalize(lines):
    return normalize_lines(lines, seq=[NFKC(), Strip()])


def preprocess_train(data, ds):
    return preprocess_pairs(data["src"]["lines"], data["trg"]["lines"], normalize_fn=normalize)


def preprocess_predict(data, ds):
    return preprocess_lines(data["lines"], normalize_fn=normalize)


def main():
    # 1. Download the corpus and write it as `train/val/test.{de,en}` files
    #    inside the AutoNMT directory layout. Tutorial 02 explains the layout.
    download_hf_dataset(
        hf_id="bentrevett/multi30k", base_path=BASE_PATH,
        dataset_name=DATASET, lang_pair=LANG_PAIR,
        src_field="de", trg_field="en",
    )

    # 2. Build the dataset: trains a SentencePiece BPE-4000 tokenizer and
    #    encodes the splits. Re-running skips stages that already exist on disk.
    builder = DatasetBuilder(
        base_path=BASE_PATH,
        datasets=[{
            "name": DATASET,
            "languages": [LANG_PAIR],
            "sizes": [("original", None)],   # "original" = the full corpus, no truncation.
        }],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [4000]}],
        preprocess_raw_fn=preprocess_train,
        preprocess_splits_fn=preprocess_train,
        merge_vocabs=False,
    ).build(force_overwrite=False)

    # The builder unrolls the cross-product (1 cell here). `get_train_ds` /
    # `get_test_ds` return the Dataset objects we iterate over.
    train_ds = builder.get_train_ds()[0]
    test_datasets = builder.get_test_ds()

    # 3. Train one tiny Transformer on the (de→en) cell.
    src_vocab, trg_vocab = train_ds.build_vocabs(max_tokens=150)
    model = Transformer.from_vocabs(src_vocab, trg_vocab)

    # `from_dataset` resolves runs_dir + run_name from the dataset variant, so
    # checkpoints and logs land under `datasets/01_hello/<...>/models/autonmt/runs/<run>`.
    trainer = AutonmtTranslator.from_dataset(
        train_ds, model=model,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        run_prefix="hello",
    )

    trainer.fit(
        train_ds,
        config=FitConfig(max_epochs=1, batch_size=128, learning_rate=1e-3, seed=42),
    )

    scores = trainer.predict(
        test_datasets,
        config=PredictConfig(
            metrics={"bleu"}, beams=[5],
            load_checkpoint="best",
            preprocess_fn=preprocess_predict,
            eval_mode="compatible",  # only score on test sets with the same lang pair
        ),
    )

    # 4. Persist the report and print a one-line summary.
    out = f".outputs/01_hello/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    _, df_summary = generate_report(scores=[scores], output_path=out)
    print(f"\nReport saved to: {os.path.abspath(out)}\n")
    print(format_summary_table(df_summary))


if __name__ == "__main__":
    main()
