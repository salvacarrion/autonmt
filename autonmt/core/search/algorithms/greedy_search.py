from autonmt.core.search.base_step_search import BaseStepSearch


class GreedySearch(BaseStepSearch):
    """Argmax decoding: at every step pick the token with the highest logit."""

    def pick_next_token(self, logits):
        return logits.argmax(dim=-1)
