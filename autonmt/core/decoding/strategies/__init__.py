"""Architecture-agnostic token-selection strategies.

A :class:`~autonmt.core.decoding.strategies.base.BaseStrategy` only knows how to
``pick_next_token`` from logits — it is reused by *both* the encoder-decoder
driver (:class:`~autonmt.core.decoding.seq2seq.step_search.StepSearch`) and the
decoder-only generator (:class:`~autonmt.core.decoding.lm.generate.LMGenerator`).
The strategies are pure: they hold no decoding loop, so this package has no
dependency on :mod:`autonmt.core.decoding.seq2seq`.
"""
from autonmt.core.decoding.strategies.base import BaseStrategy
from autonmt.core.decoding.strategies.greedy import GreedySearch
from autonmt.core.decoding.strategies.multinomial import MultinomialSampling
from autonmt.core.decoding.strategies.topk import TopKSampling
from autonmt.core.decoding.strategies.topp import TopPSampling

__all__ = ["BaseStrategy", "GreedySearch", "MultinomialSampling",
           "TopKSampling", "TopPSampling"]
