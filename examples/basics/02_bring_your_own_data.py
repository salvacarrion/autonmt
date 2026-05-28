"""
============================================================================
 Tutorial 02 — Bring your own data
============================================================================

What you'll learn
-----------------
The on-disk layout AutoNMT relies on. Every Translator and the DatasetBuilder
are path-driven: stages live in numbered subfolders and re-runs skip stages
that are already materialised.

Expected layout for a dataset variant
-------------------------------------
    <base_path>/<dataset_name>/<src-trg>/<size>/
    ├── data/
    │   ├── 0_raw/                    (optional)   <- a parallel corpus pair: data.<src>, data.<trg>
    │   ├── 1_splits/                              <- train.<lang>, val.<lang>, test.<lang>
    │   ├── 2_preprocessed/                        <- written by preprocess_splits_fn
    │   ├── 3_pretokenized/                        <- only for `word` subword (Moses)
    │   └── 4_encoded/<subword>/<vocab_size>/      <- SPM-encoded splits
    ├── models/<engine>/runs/<run>/                <- checkpoints, logs, evals
    ├── stats/                                     <- corpus statistics (lengths, OOVs, ...)
    └── vocabs/                                    <- the SPM model and tab-separated vocab

You only need to provide ONE of `0_raw/` or `1_splits/`. If you give raw, the
builder splits it; if you give pre-split files, it uses them directly. Files
use the language as the *extension* (`train.es`, never `es/train.txt`).

What's new vs tutorial 01
-------------------------
Tutorial 01 used `download_hf_dataset` to materialise multi30k for you. Here we
*don't* download anything — we point the builder at files we wrote ourselves.
To make the script self-runnable, a small `bootstrap_local_data` helper writes
toy parallel files into the expected layout on first run. In a real project
you would replace that helper with your own data.

Run
---
    python examples/basics/02_bring_your_own_data.py
"""
import datetime
import os

from tokenizers.normalizers import NFKC, Strip

from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.nn.models import Transformer
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.processors import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.reporting.report import format_summary_table, generate_report

BASE_PATH = "datasets/02_byo"
DATASET = "tiny_corpus"
LANG_PAIR = "es-en"
SIZE = "original"


def bootstrap_local_data():
    """Create `<BASE_PATH>/tiny_corpus/es-en/original/data/1_splits/` with toy files.

    Real users replace this with whatever produces their own train/val/test
    files. AutoNMT does not care how the files got there; only that they exist
    at the expected paths.
    """
    splits_dir = os.path.join(BASE_PATH, DATASET, LANG_PAIR, SIZE, "data", "1_splits")
    if os.path.exists(os.path.join(splits_dir, "train.es")):
        return  # already bootstrapped on a previous run.

    os.makedirs(splits_dir, exist_ok=True)
    pairs = [
        ("Hola, ¿cómo estás?", "Hello, how are you?"),
        ("Buenos días.", "Good morning."),
        ("Me llamo Ana.", "My name is Ana."),
        ("¿Dónde está la biblioteca?", "Where is the library?"),
        ("Tengo hambre.", "I am hungry."),
        ("Hace mucho calor hoy.", "It is very hot today."),
        ("Mañana iré al parque.", "Tomorrow I will go to the park."),
        ("Me gusta leer libros.", "I like reading books."),
        ("Estoy aprendiendo español.", "I am learning Spanish."),
        ("El gato duerme en el sofá.", "The cat sleeps on the couch."),
    ] * 200  # repeat so SPM has enough material to train on

    train, val, test = pairs[:1600], pairs[1600:1800], pairs[1800:2000]
    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        for lang_idx, lang in enumerate(["es", "en"]):
            with open(os.path.join(splits_dir, f"{split_name}.{lang}"), "w", encoding="utf-8") as f:
                f.write("\n".join(row[lang_idx] for row in split) + "\n")
    print(f"[bootstrap] Toy corpus written to: {splits_dir}")


def normalize(lines):
    return normalize_lines(lines, seq=[NFKC(), Strip()])


def preprocess_train(data, ds):
    return preprocess_pairs(data["src"]["lines"], data["trg"]["lines"], normalize_fn=normalize)


def preprocess_predict(data, ds):
    return preprocess_lines(data["lines"], normalize_fn=normalize)


def main():
    bootstrap_local_data()

    # Same DatasetBuilder call as tutorial 01 — the only difference is that the
    # files at `<BASE_PATH>/<dataset>/<lang>/<size>/data/1_splits/` were written
    # by us, not by `download_hf_dataset`.
    builder = DatasetBuilder(
        base_path=BASE_PATH,
        datasets=[{
            "name": DATASET,
            "languages": [LANG_PAIR],
            "sizes": [(SIZE, None)],
        }],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [500]}],  # tiny vocab for a tiny corpus
        preprocess_raw_fn=preprocess_train,
        preprocess_splits_fn=preprocess_train,
        merge_vocabs=False,
    ).build(force_overwrite=False)

    train_ds = builder.get_train_ds()[0]
    test_datasets = builder.get_test_ds()

    src_vocab, trg_vocab = train_ds.build_vocabs(max_tokens=150)
    model = Transformer.from_vocabs(src_vocab, trg_vocab)

    trainer = AutonmtTranslator.from_dataset(
        train_ds, model=model,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        run_prefix="byo",
    )

    trainer.fit(
        train_ds,
        config=FitConfig(max_epochs=2, batch_size=32, learning_rate=1e-3, seed=42),
    )

    scores = trainer.predict(
        test_datasets,
        config=PredictConfig(
            metrics={"bleu"}, beams=[5],
            load_checkpoint="best",
            preprocess_fn=preprocess_predict,
            eval_mode="compatible",
        ),
    )

    out = f".outputs/02_byo/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    _, df_summary = generate_report(scores=[scores], output_path=out)
    print(f"\nReport saved to: {os.path.abspath(out)}\n")
    print(format_summary_table(df_summary))


if __name__ == "__main__":
    main()
