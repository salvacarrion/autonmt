"""The torch ``Dataset`` for masked language modelling (packed blocks + dynamic masking).

Like :class:`~autonmt.core.data.lm_dataset.LMDataset` it reads a flat, packed
token stream produced by :class:`~autonmt.datasets.lm.lm_corpus.LMCorpusBuilder`
(``mode="mlm"``) and slices it into fixed ``block_size`` windows. But the MLM
objective is **not** next-token prediction: there is no shift. Instead, each
item corrupts a random subset of the block and asks the model to recover the
originals at those positions.

Masking is **dynamic** — re-sampled on every ``__getitem__`` — so each epoch sees
a different corruption of the same data (RoBERTa-style), which is why it lives
here rather than being baked into the packed file.

Each item is ``(x, y)``:
  * ``x`` — the corrupted block. ~``mlm_prob`` of the (non-special) positions are
    replaced following the BERT 80/10/10 scheme: 80% with ``<mask>``, 10% with a
    random token, 10% left unchanged.
  * ``y`` — the targets: the **original** token at the selected positions and
    ``ignore_index`` everywhere else, so only the corrupted positions contribute
    to the loss.

References
----------
Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding.* (the masked-LM objective and 80/10/10 scheme)
[arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.*
(dynamic masking re-sampled per epoch)
[arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class MLMDataset(Dataset):
    def __init__(self, tokens_file, block_size, mask_id, vocab_size, special_ids,
                 mlm_prob=0.15, ignore_index=-100):
        self.tokens = np.load(tokens_file, mmap_mode="r")
        self.block_size = int(block_size)
        self.mask_id = int(mask_id)
        self.vocab_size = int(vocab_size)
        self.special_ids = set(int(i) for i in special_ids)
        self.mlm_prob = float(mlm_prob)
        self.ignore_index = int(ignore_index)

        # No shift (unlike the causal LM), so a block is exactly block_size tokens.
        self.n_blocks = len(self.tokens) // self.block_size
        if self.n_blocks < 1:
            raise ValueError(
                f"Corpus too small for block_size={self.block_size}: need at least "
                f"{self.block_size} tokens, got {len(self.tokens)}."
            )

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        s = idx * self.block_size
        block = np.asarray(self.tokens[s:s + self.block_size], dtype=np.int64)
        x = torch.from_numpy(block.copy())
        y = torch.full((self.block_size,), self.ignore_index, dtype=torch.long)

        # Candidate positions: everything except special tokens (sos/eos/pad/unk/mask).
        candidates = torch.tensor(
            [i for i, t in enumerate(block) if int(t) not in self.special_ids],
            dtype=torch.long,
        )
        if candidates.numel() == 0:
            return x, y

        n_pred = max(1, int(round(candidates.numel() * self.mlm_prob)))
        chosen = candidates[torch.randperm(candidates.numel())[:n_pred]]

        # Targets are the originals at the chosen positions; loss ignores the rest.
        y[chosen] = x[chosen]

        # BERT 80/10/10 corruption of the chosen positions.
        r = torch.rand(chosen.numel())
        mask_sel = chosen[r < 0.8]
        rand_sel = chosen[(r >= 0.8) & (r < 0.9)]
        x[mask_sel] = self.mask_id
        if rand_sel.numel() > 0:
            x[rand_sel] = torch.randint(0, self.vocab_size, (rand_sel.numel(),), dtype=torch.long)
        # The remaining ~10% are left unchanged (still supervised via y).
        return x, y
