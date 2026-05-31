import torch

from autonmt.core.decoding.base_step_search import BaseStepSearch


class TopKSampling(BaseStepSearch):
    """Top-k sampling.

    At every step, keep the ``top_k`` tokens with the highest logits, zero-out
    the rest, renormalize and sample. ``temperature`` is applied to the logits
    before the top-k filter — same semantics as :class:`MultinomialSampling`.

    Output is non-deterministic — seed ``torch.manual_seed`` for reproducibility.

    References
    ----------
    Fan, Lewis & Dauphin (2018). *Hierarchical Neural Story Generation.*
    [arXiv:1805.04833](https://arxiv.org/abs/1805.04833)
    """

    def __init__(self, top_k=50, temperature=1.0):
        assert top_k >= 1
        assert temperature > 0
        self.top_k = top_k
        self.temperature = temperature

    def pick_next_token(self, logits):
        if self.temperature != 1.0:
            logits = logits / self.temperature

        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        probs = topk_vals.softmax(dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)             # (B, 1) in topk-space
        return topk_idx.gather(-1, sampled).squeeze(-1)               # map back to vocab ids
