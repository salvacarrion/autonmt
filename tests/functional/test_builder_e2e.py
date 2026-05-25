"""End-to-end test of the preprocessing pipeline against a synthetic tiny corpus.

We deliberately avoid Torch / Lightning so this can run in CI with only the
preprocessing extras installed. The test covers:

  1. Raw → splits materialisation
  2. Tokenizer training (SentencePiece BPE)
  3. Subword encoding of each split

If SentencePiece isn't available the test is skipped, not failed.
"""
import os
import shutil
import tempfile

import pytest

pytest.importorskip("sentencepiece")
pytest.importorskip("sacremoses")
pytest.importorskip("tokenizers")

from autonmt.datasets.dataset_builder import DatasetBuilder  # noqa: E402


SRC_SENTENCES = [
    "the cat sat on the mat .",
    "a quick brown fox jumps over the lazy dog .",
    "machine translation is fun .",
    "this is a simple test .",
    "we are training a small model .",
    "the weather is nice today .",
    "i like programming in python .",
    "neural networks learn from data .",
    "the book is on the table .",
    "she went to the store yesterday .",
] * 60  # 600 lines — enough for SentencePiece to train

TRG_SENTENCES = [
    "el gato se sento en la alfombra .",
    "un rapido zorro marron salta sobre el perro perezoso .",
    "la traduccion automatica es divertida .",
    "esta es una prueba simple .",
    "estamos entrenando un modelo pequeno .",
    "el clima es agradable hoy .",
    "me gusta programar en python .",
    "las redes neuronales aprenden de los datos .",
    "el libro esta sobre la mesa .",
    "ella fue a la tienda ayer .",
] * 60


@pytest.fixture
def tiny_corpus():
    """Materialise a synthetic en-es corpus in the AutoNMT layout and yield its base."""
    tmp = tempfile.mkdtemp(prefix="autonmt_e2e_")
    try:
        raw_dir = os.path.join(tmp, "tiny", "en-es", "original", "data", "0_raw")
        os.makedirs(raw_dir, exist_ok=True)
        with open(os.path.join(raw_dir, "data.en"), "w") as f:
            f.write("\n".join(SRC_SENTENCES) + "\n")
        with open(os.path.join(raw_dir, "data.es"), "w") as f:
            f.write("\n".join(TRG_SENTENCES) + "\n")
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_builder_runs_full_pipeline(tiny_corpus):
    builder = DatasetBuilder(
        base_path=tiny_corpus,
        datasets=[{
            "name": "tiny",
            "languages": ["en-es"],
            "sizes": [("original", None)],
            "split_sizes": (None, 50, 50),  # small enough for a 600-line corpus
        }],
        encoding=[{
            "subword_models": ["bpe"],
            "vocab_sizes": [200],  # small but viable for SentencePiece
        }],
        merge_vocabs=False,
    ).build(force_overwrite=False)

    train_ds = builder.get_train_ds()
    assert len(train_ds) == 1
    ds = train_ds[0]

    # Splits got materialised
    for fname in ds.get_split_fnames():
        assert os.path.exists(ds.get_split_path(fname)), f"missing split file: {fname}"

    # SentencePiece trained per-language (merge_vocabs=False)
    for lang in ("en", "es"):
        vocab_file = ds.get_vocab_file(lang=lang)
        assert os.path.exists(vocab_file + ".model"), f"missing SP model: {lang}"
        assert os.path.exists(vocab_file + ".vocab"), f"missing SP vocab: {lang}"

    # Encoded files written
    for fname in ds.get_split_fnames():
        assert os.path.exists(ds.get_encoded_path(fname)), f"missing encoded file: {fname}"

    # Re-running with force_overwrite=False is a no-op (idempotent stage skipping)
    DatasetBuilder(
        base_path=tiny_corpus,
        datasets=[{
            "name": "tiny",
            "languages": ["en-es"],
            "sizes": [("original", None)],
            "split_sizes": (None, 50, 50),
        }],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [200]}],
        merge_vocabs=False,
    ).build(force_overwrite=False)
