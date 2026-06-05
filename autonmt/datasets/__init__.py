"""Public dataset API.

Organized by role: ``parallel/`` (translation datasets), ``lm/`` (single-stream
corpora), ``processing/`` (preprocess / encode / tokenize), ``sources/`` (loaders);
``stats`` and ``leakage`` are cross-cutting helpers. The headline builders and
identity classes are re-exported here so ``from autonmt.datasets import DatasetBuilder``
stays the entry point.
"""
from autonmt.datasets.parallel.dataset import Dataset, DatasetLayout
from autonmt.datasets.parallel.dataset_builder import DatasetBuilder, merge_datasets
from autonmt.datasets.lm.lm_corpus import LMCorpus, LMCorpusBuilder

__all__ = ["Dataset", "DatasetLayout", "DatasetBuilder", "merge_datasets",
           "LMCorpus", "LMCorpusBuilder"]
