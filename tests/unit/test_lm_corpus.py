"""End-to-end build of LM corpora (text + instruct) on a tiny synthetic stream.

Hermetic: no network, no on-disk fixtures — inline ``text`` / ``pairs`` are
written to ``0_raw`` by the builder, SentencePiece is trained, and the packed
token streams are produced. Mirrors the project's synthetic-corpus E2E style.
"""
import os
import random

import numpy as np
import pytest

from autonmt.datasets.lm_corpus import LMCorpusBuilder, TEXT, INSTRUCT


WORDS = ("the quick brown fox jumps over lazy dog cat runs fast slow under "
         "bright moon sky river stone green blue red").split()


def _sentence(rng):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 12)))


def _build(tmp_path, mode):
    rng = random.Random(0)
    if mode == TEXT:
        decl = {"name": "tt", "mode": TEXT, "sizes": [("original", None)],
                "text": [_sentence(rng) for _ in range(200)]}
    else:
        decl = {"name": "qa", "mode": INSTRUCT, "sizes": [("original", None)],
                "pairs": [(f"q: {_sentence(rng)}", _sentence(rng)) for _ in range(150)]}
    builder = LMCorpusBuilder(
        base_path=str(tmp_path), corpus=[decl],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [120]}],
        val_size=0.1,
    ).build(force_overwrite=True)
    return builder.get_train_ds()[0]


def test_build_text_corpus(tmp_path):
    corpus = _build(tmp_path, TEXT)
    assert corpus.mode == TEXT
    assert os.path.exists(corpus.spm_model_path())
    assert corpus.model_vocab_size == 120

    for split in corpus.splits:
        tokens = np.load(corpus.tokens_file(split))
        assert tokens.ndim == 1 and len(tokens) > 0
        # Pure-LM corpora carry no supervise mask.
        assert not os.path.exists(corpus.supervise_file(split))


def test_build_instruct_corpus(tmp_path):
    corpus = _build(tmp_path, INSTRUCT)
    assert corpus.mode == INSTRUCT
    for split in corpus.splits:
        tokens = np.load(corpus.tokens_file(split))
        supervise = np.load(corpus.supervise_file(split))
        assert len(tokens) == len(supervise)
        # Some positions are masked (prompt) and some supervised (completion).
        assert supervise.min() == 0 and supervise.max() == 1


def test_encode_decode_roundtrip(tmp_path):
    corpus = _build(tmp_path, TEXT)
    ids = corpus.encode("the quick brown fox", add_sos=True, add_eos=True)
    assert ids[0] == corpus.sos_id and ids[-1] == corpus.eos_id
    # Special tokens are stripped on decode; round-trip recovers the surface form.
    assert corpus.decode(ids) == "the quick brown fox"


def test_rejects_non_sentencepiece_model(tmp_path):
    with pytest.raises(ValueError, match="SentencePiece"):
        LMCorpusBuilder(
            base_path=str(tmp_path),
            corpus=[{"name": "x", "mode": TEXT, "sizes": [("original", None)], "text": ["a b c"]}],
            encoding=[{"subword_models": ["bytes"], "vocab_sizes": [120]}],
        )
