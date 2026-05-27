import torch

from autonmt.core.decoding.base_step_search import BaseStepSearch


class TopPSampling(BaseStepSearch):
    """Top-p sampling (a.k.a. *nucleus* sampling, Holtzman et al. 2019).

    At every step, keep the smallest set of tokens whose cumulative probability
    is at least ``top_p`` (the "nucleus"), zero-out the rest, and sample from
    the renormalized distribution. ``temperature`` is applied to the logits
    before nucleus selection — same semantics as :class:`MultinomialSampling`.

    Output is non-deterministic — seed ``torch.manual_seed`` for reproducibility.
    """

    def __init__(self, top_p=0.9, temperature=1.0):
        assert 0.0 < top_p <= 1.0
        assert temperature > 0
        self.top_p = top_p
        self.temperature = temperature

    def pick_next_token(self, logits):
        if self.temperature != 1.0:
            logits = logits / self.temperature

        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Tokens *strictly* above top_p are masked, but we shift the mask one
        # slot to the right so that the boundary token (the one that pushed the
        # cumulative mass over top_p) is kept — guarantees at least one token.
        remove = cum_probs > self.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        probs = sorted_logits.softmax(dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)             # (B, 1) in sorted-space
        return sorted_idx.gather(-1, sampled).squeeze(-1)             # map back to vocab ids
