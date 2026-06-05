"""Decoder-only (causal LM) generation.

The decoder-only counterpart of :mod:`autonmt.core.decoding.seq2seq`: same role
(drive the per-step loop) but prompt-prefill into the KV cache instead of
encoder-seeded. Reuses the :mod:`autonmt.core.decoding.strategies` strategies.
"""
from autonmt.core.decoding.lm.generate import LMGenerator

__all__ = ["LMGenerator"]
