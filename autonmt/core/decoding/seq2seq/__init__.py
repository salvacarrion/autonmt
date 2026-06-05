"""Encoder-decoder (seq2seq) decoders.

The drivers here are *encoder-seeded*: they call ``forward_encoder`` and start
from a single ``<sos>``, sized relative to the source length. Not interchangeable
with the decoder-only :mod:`autonmt.core.decoding.lm` generator — hence the split.
``StepSearch`` composes a :mod:`autonmt.core.decoding.strategies` strategy for the
per-step token choice; ``BeamSearch`` owns its own beam loop.
"""
from autonmt.core.decoding.seq2seq.base_search import BaseSearch
from autonmt.core.decoding.seq2seq.step_search import StepSearch
from autonmt.core.decoding.seq2seq.beam_search import BeamSearch

__all__ = ["BaseSearch", "StepSearch", "BeamSearch"]
