"""Masking correctness for the MLM dataset.

The masked-LM contract:
  * each item is exactly ``block_size`` long (no shift, unlike the causal LM),
  * a subset of non-special positions is selected; targets there are the
    ORIGINAL tokens, everything else is ``ignore_index``,
  * special tokens are never selected as targets,
  * non-target input positions are left unchanged.
"""
import numpy as np
import pytest
import torch

from autonmt.core.data.mlm_dataset import MLMDataset


def _save(tmp_path, name, arr):
    path = str(tmp_path / name)
    np.save(path, np.asarray(arr, dtype=np.uint16))
    return path


def _ds(tmp_path, tokens, block_size=8, mask_id=4, vocab_size=50, special_ids=(0, 1, 2, 3, 4),
        mlm_prob=0.5, ignore_index=3):
    path = _save(tmp_path, "tokens.npy", tokens)
    return MLMDataset(path, block_size=block_size, mask_id=mask_id, vocab_size=vocab_size,
                      special_ids=special_ids, mlm_prob=mlm_prob, ignore_index=ignore_index)


def test_block_count_no_shift(tmp_path):
    # 20 tokens, block_size 8 -> floor(20/8) = 2 blocks (no +1, MLM doesn't shift).
    ds = _ds(tmp_path, list(range(5, 25)), block_size=8)
    assert len(ds) == 2
    x, y = ds[0]
    assert x.shape == (8,) and y.shape == (8,)


def test_targets_and_ignore(tmp_path):
    torch.manual_seed(0)
    tokens = [10, 11, 12, 13, 14, 15, 16, 17]  # all non-special
    ds = _ds(tmp_path, tokens, block_size=8, mlm_prob=0.5, ignore_index=3)
    x, y = ds[0]
    original = torch.tensor(tokens)

    predicted = (y != 3)
    assert predicted.sum() >= 1                       # at least one target
    # Targets carry the ORIGINAL token; non-targets are ignore_index.
    assert torch.equal(y[predicted], original[predicted])
    # Non-target input positions are untouched.
    assert torch.equal(x[~predicted], original[~predicted])


def test_special_tokens_never_targets(tmp_path):
    torch.manual_seed(0)
    # Interleave specials (ids 0-4) with real tokens; specials must never be predicted.
    tokens = [1, 10, 2, 11, 0, 12, 4, 13, 1, 14, 2, 15, 3, 16, 0, 17]
    ds = _ds(tmp_path, tokens, block_size=16, special_ids=(0, 1, 2, 3, 4), mlm_prob=1.0,
             ignore_index=3)
    x, y = ds[0]
    original = torch.tensor(tokens)
    special_pos = torch.tensor([t in (0, 1, 2, 3, 4) for t in tokens])
    # No special position is a target.
    assert (y[special_pos] == 3).all()
    # With mlm_prob=1.0 every non-special position IS a target.
    assert (y[~special_pos] == original[~special_pos]).all()


def test_too_small_raises(tmp_path):
    with pytest.raises(ValueError, match="too small"):
        _ds(tmp_path, [1, 2, 3], block_size=8)
