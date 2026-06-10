"""Single-stream LM corpora: the ``LMCorpus`` identity/path holder (``corpus.py``)
and the ``LMCorpusBuilder`` (``builder.py``). Single-stream sibling of
:mod:`autonmt.datasets.parallel`.
"""
from autonmt.datasets.lm.corpus import LMCorpus, TEXT, INSTRUCT, MLM
from autonmt.datasets.lm.builder import LMCorpusBuilder

__all__ = ["LMCorpus", "LMCorpusBuilder", "TEXT", "INSTRUCT", "MLM"]
