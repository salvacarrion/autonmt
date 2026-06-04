# Datasets

Corpus preparation: the builder, the dataset/identity holder, and the
subword-agnostic / subword-dependent preprocessing layers.

## DatasetBuilder

::: autonmt.datasets.dataset_builder.DatasetBuilder

## Dataset

::: autonmt.datasets.dataset.Dataset

## LMCorpusBuilder

The single-stream / instruct sibling of `DatasetBuilder`. See
[LM corpora](../guide/data/lm-corpora.md).

::: autonmt.datasets.lm_corpus.LMCorpusBuilder

## LMCorpus

::: autonmt.datasets.lm_corpus.LMCorpus

## Preprocessing

Subword-agnostic cleanup: filter, normalize, dedupe.

::: autonmt.datasets.preprocessing

## Encoding

Subword-dependent: Moses pretokenize, SentencePiece / bytes encode & decode.

::: autonmt.datasets.encoding

## HuggingFace loader

::: autonmt.datasets.hf_loader

## Leakage checks

::: autonmt.datasets.leakage

## Statistics

::: autonmt.datasets.stats
