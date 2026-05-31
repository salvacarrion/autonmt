import torch
from torch.utils.data import BatchSampler
import numpy as np


class BucketSampler(BatchSampler):
    """Length-bucketed batch sampler.

    Groups examples by length and emits batches either of fixed size
    (`batch_size`) or of variable size under a token budget (`max_tokens`).
    Exactly one of `batch_size` / `max_tokens` must be provided.

    Bucket composition is precomputed once; on each new iteration only the
    *order* in which batches are emitted is reshuffled, with a per-epoch
    seed so the model sees a different sequence every epoch.

    References
    ----------
    Vaswani et al. (2017). *Attention Is All You Need.* (length-based batching
    and per-batch token budgets, §5.1)
    [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
    """

    def __init__(self, data_source, sort_key, batch_size=None, max_tokens=None,
                 shuffle=True, sort_within_batch=False, seed=0):
        assert (batch_size is None) != (max_tokens is None), \
            "Specify exactly one of batch_size or max_tokens"
        self.data_source = data_source
        self.sort_key = sort_key
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.sort_within_batch = sort_within_batch
        self.seed = seed
        self.epoch = 0

        self.lengths = np.array(
            [sort_key(x, y) for x, y in data_source], dtype=np.int64
        )
        sorted_idx = np.argsort(self.lengths, kind="stable")
        self.batches = self._build_batches(sorted_idx)

    def _build_batches(self, sorted_idx):
        if self.batch_size is not None:
            return [sorted_idx[i:i + self.batch_size].tolist()
                    for i in range(0, len(sorted_idx), self.batch_size)]

        # max_tokens: greedily pack until (n+1)*max_len would exceed the budget.
        batches, current, current_max = [], [], 0
        for idx in sorted_idx:
            idx = int(idx)
            ex_len = int(self.lengths[idx])
            assert ex_len <= self.max_tokens, \
                f"Example {idx} has length {ex_len} > max_tokens {self.max_tokens}"
            new_max = max(current_max, ex_len)
            if current and (len(current) + 1) * new_max > self.max_tokens:
                batches.append(current)
                current, current_max = [idx], ex_len
            else:
                current.append(idx)
                current_max = new_max
        if current:
            batches.append(current)
        return batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        batches = self.batches
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.epoch += 1
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]
        if self.sort_within_batch:
            batches = [sorted(b, key=lambda i: -int(self.lengths[i]))
                       for b in batches]
        yield from batches

    def __len__(self):
        return len(self.batches)
