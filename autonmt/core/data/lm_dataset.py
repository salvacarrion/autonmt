"""The torch ``Dataset`` for single-stream language modelling (packed blocks).

Unlike :class:`~autonmt.core.data.translation_dataset.TranslationDataset` (one
padded sentence pair per item), this reads a *flat* token-id stream produced by
:class:`~autonmt.datasets.lm.builder.LMCorpusBuilder`, concatenated across
documents, and slices it into fixed ``block_size`` windows (nanoGPT-style
packing). No padding, no bucketing — every item is exactly ``block_size`` long,
so a plain ``DataLoader`` with the default collate stacks them into
``(batch, block_size)``.

Each item is ``(x, y, supervise)``:
  * ``x = block[:-1]`` — the input tokens.
  * ``y = block[1:]`` — the next-token targets.
  * ``supervise`` — 1 where the position contributes to the loss, 0 where it is
    ignored. For pure LM every position is supervised; for instruct corpora the
    prompt span is masked out (see :class:`LMCorpusBuilder`), so the model only
    learns to predict the completion.

References
----------
Brown et al. (2020). *Language Models are Few-Shot Learners.* (packing the token
stream into contiguous fixed-length training blocks)
[arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class LMDataset(Dataset):
    def __init__(self, tokens_file, block_size, supervise_file=None):
        # mmap so a multi-GB packed corpus isn't pulled into RAM up-front; each
        # __getitem__ copies only its own block_size+1 window.
        self.tokens = np.load(tokens_file, mmap_mode="r")
        self.supervise = (np.load(supervise_file, mmap_mode="r")
                          if supervise_file is not None else None)
        self.block_size = int(block_size)

        # Each block needs block_size inputs + 1 shifted target → block_size+1
        # tokens. The trailing remainder (< block_size+1) is dropped, as is
        # standard for packed LM training.
        self.n_blocks = (len(self.tokens) - 1) // self.block_size
        if self.n_blocks < 1:
            raise ValueError(
                f"Corpus too small for block_size={self.block_size}: need at least "
                f"{self.block_size + 1} tokens, got {len(self.tokens)}. Use a smaller "
                f"block_size or a larger corpus."
            )

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        s = idx * self.block_size
        # torch has no uint16/uint32 dtype, so cast the packed ids to int64 here.
        chunk = np.asarray(self.tokens[s:s + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:].copy())

        if self.supervise is not None:
            # supervise[t] marks whether target position t counts for the loss;
            # align it with y (the +1 shift) and take the same window as y.
            sup = np.asarray(self.supervise[s + 1:s + self.block_size + 1], dtype=np.int64)
            supervise = torch.from_numpy(sup)
        else:
            supervise = torch.ones(self.block_size, dtype=torch.long)

        return x, y, supervise
