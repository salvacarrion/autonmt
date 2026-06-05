from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Architecture-agnostic token-selection strategy.

    The one piece of decoding that does *not* depend on whether the model is
    encoder-decoder or decoder-only: given the logits at the current position,
    choose the next token. The seq2seq driver
    (:class:`~autonmt.core.decoding.seq2seq.step_search.StepSearch`) and the
    decoder-only :class:`~autonmt.core.decoding.lm.generate.LMGenerator` each
    drive their *own* loop — encoder-seeded vs prompt-prefill — and delegate the
    per-step choice to a strategy, so greedy / top-k / top-p / multinomial are
    written once here and reused by both families.
    """

    @abstractmethod
    def pick_next_token(self, logits):
        """Choose the next token per sequence.

        ``logits`` has shape ``(B, V)`` (unnormalized scores at the current
        position). Return a ``LongTensor`` of shape ``(B,)`` with the chosen
        token id per row.
        """
        ...
