# Datasets

Corpus preparation, in two tiers: the **corpus types** (`parallel/`, `lm/`) —
each an identity/path holder plus a cross-product builder — and the **machinery**
that operates on any corpus (`processing/`, `sources/`, `analysis/`).

## ParallelCorpusBuilder

::: autonmt.datasets.parallel.builder.ParallelCorpusBuilder

### merge_corpora

::: autonmt.datasets.parallel.builder.merge_corpora

## ParallelCorpus

The per-cell identity + path engine. `ParallelCorpus` extends
`ParallelCorpusLayout` (the pure path computer) with builder-time state and
disk-inspection helpers.

::: autonmt.datasets.parallel.corpus.ParallelCorpus

::: autonmt.datasets.parallel.corpus.ParallelCorpusLayout

## LMCorpusBuilder

The single-stream / instruct sibling of `ParallelCorpusBuilder`. See
[LM corpora](../guide/data/lm-corpora.md).

::: autonmt.datasets.lm.builder.LMCorpusBuilder

## LMCorpus

::: autonmt.datasets.lm.corpus.LMCorpus

## Preprocessing

Subword-agnostic cleanup: filter, normalize, dedupe.

::: autonmt.datasets.processing.preprocessing

## Encoding

Subword-dependent: Moses pretokenize, SentencePiece / bytes encode & decode.

::: autonmt.datasets.processing.encoding

## Splits

Split-preparation helpers: resolve a variant's size spec and co-shuffle the two
sides of a parallel split in lockstep.

::: autonmt.datasets.processing.splits

## HuggingFace loader

::: autonmt.datasets.sources.hf_loader

## Leakage checks

::: autonmt.datasets.analysis.leakage

## Statistics

::: autonmt.datasets.analysis.stats
