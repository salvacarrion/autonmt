"""Packing correctness for the LM dataset.

The packed-block contract:
  * each item is exactly ``block_size`` long (no padding),
  * ``y`` is ``x`` shifted by one within the block,
  * the ``supervise`` window aligns with ``y`` (it is the +1-shifted slice),
  * pure-LM (no supervise file) supervises every position.
"""
import numpy as np
import pytest
import torch

from autonmt.core.data.lm_dataset import LMDataset


def _save(tmp_path, name, arr, dtype=np.uint16):
    path = str(tmp_path / name)  # name ends in .npy → np.save writes it verbatim
    np.save(path, np.asarray(arr, dtype=dtype))
    return path


def test_blocks_and_shift(tmp_path):
    tokens = list(range(21))  # 0..20
    path = _save(tmp_path, "tokens.npy", tokens)
    ds = LMDataset(path, block_size=4)

    assert len(ds) == (21 - 1) // 4  # == 5
    x, y, sup = ds[0]
    assert x.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [1, 2, 3, 4]
    assert torch.equal(y[:-1], x[1:])      # y is x shifted by one
    assert sup.tolist() == [1, 1, 1, 1]    # pure LM: every position supervised

    # Second block starts where the first's inputs ended (non-overlapping stride).
    x1, y1, _ = ds[1]
    assert x1.tolist() == [4, 5, 6, 7]


def test_supervise_alignment(tmp_path):
    tokens = [10, 11, 12, 13, 14, 15]
    supervise = [0, 0, 1, 1, 1, 1]  # first two tokens are "prompt"
    tok_path = _save(tmp_path, "tok.npy", tokens)
    sup_path = _save(tmp_path, "sup.npy", supervise, dtype=np.uint8)
    ds = LMDataset(tok_path, block_size=2, supervise_file=sup_path)

    x, y, sup = ds[0]
    assert x.tolist() == [10, 11]
    assert y.tolist() == [11, 12]
    # supervise aligns with y == supervise[1:3]
    assert sup.tolist() == [0, 1]


def test_too_small_corpus_raises(tmp_path):
    path = _save(tmp_path, "tiny.npy", [1, 2, 3])
    with pytest.raises(ValueError, match="too small"):
        LMDataset(path, block_size=5)
