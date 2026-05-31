from autonmt.core.decoding.base_step_search import BaseStepSearch


class GreedySearch(BaseStepSearch):
    """Argmax (greedy) decoding.

    At every step pick the token with the highest logit. Deterministic and
    fast; equivalent to beam search with ``beam_width=1``.
    """

    def pick_next_token(self, logits):
        return logits.argmax(dim=-1)
