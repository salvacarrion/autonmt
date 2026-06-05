"""Decoding, organized by responsibility.

* :mod:`~autonmt.core.decoding.strategies` — architecture-agnostic token pickers
  (the :class:`BaseStrategy` everyone reuses): greedy / top-k / top-p / multinomial.
* :mod:`~autonmt.core.decoding.seq2seq` — encoder-decoder drivers (encoder-seeded
  ``BaseSearch`` / ``StepSearch`` / ``BeamSearch``).
* :mod:`~autonmt.core.decoding.lm` — decoder-only generation (``LMGenerator``,
  prompt-prefill).

A *strategy* (how to pick the next token) is composed into a *driver* (the loop):
``StepSearch(TopPSampling())`` for seq2seq, ``LMGenerator(TopPSampling())`` for LMs.
"""
from autonmt.core.decoding.strategies import (
    BaseStrategy,
    GreedySearch,
    MultinomialSampling,
    TopKSampling,
    TopPSampling,
)
from autonmt.core.decoding.seq2seq import BaseSearch, StepSearch, BeamSearch
from autonmt.core.decoding.lm import LMGenerator

__all__ = [
    "BaseStrategy",
    "GreedySearch", "MultinomialSampling", "TopKSampling", "TopPSampling",
    "BaseSearch", "StepSearch", "BeamSearch",
    "LMGenerator",
]
