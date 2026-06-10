"""Public corpus API.

Two tiers live under here. **Corpus types** (the nouns): ``parallel/``
(translation, ``xx-yy``-keyed) and ``lm/`` (single-stream / instruct corpora),
each a ``corpus.py`` + ``builder.py`` pair. **Machinery** that operates on any
corpus: ``processing/`` (preprocess / encode / tokenize / split), ``sources/``
(loaders), and ``analysis/`` (descriptive stats + leakage detection).

The headline corpus + builder classes are re-exported here, so
``from autonmt.datasets import ParallelCorpusBuilder`` stays the entry point.
"""
from autonmt.datasets.parallel.corpus import ParallelCorpus, ParallelCorpusLayout
from autonmt.datasets.parallel.builder import ParallelCorpusBuilder, merge_corpora
from autonmt.datasets.lm.corpus import LMCorpus
from autonmt.datasets.lm.builder import LMCorpusBuilder

__all__ = ["ParallelCorpus", "ParallelCorpusLayout", "ParallelCorpusBuilder",
           "merge_corpora", "LMCorpus", "LMCorpusBuilder"]
