import torch

from autonmt.core.decoding.base_step_search import BaseStepSearch


class MultinomialSampling(BaseStepSearch):
    """Temperature-scaled multinomial sampling.

    ``temperature`` rescales the logits before the softmax:
      * ``< 1`` sharpens the distribution (closer to argmax / greedy).
      * ``= 1`` is the model's native distribution.
      * ``> 1`` flattens it (closer to uniform).
    Output is non-deterministic — seed ``torch.manual_seed`` for reproducibility.
    """

    def __init__(self, temperature=1.0):
        assert temperature > 0
        self.temperature = temperature

    def pick_next_token(self, logits):
        if self.temperature != 1.0:
            logits = logits / self.temperature
        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
