# Models & decoding

AutoNMT's in-house neural engine: the Lightning base class, built-in architectures,
decoding strategies, samplers, and the torch dataset.

## LitSeq2Seq

The base every custom model inherits. See [Custom models](../guides/custom-models.md).

::: autonmt.core.nn.seq2seq.LitSeq2Seq

## Built-in models

::: autonmt.core.nn.models

## Decoding strategies

::: autonmt.core.decoding

## Translation dataset

The torch `Dataset` used internally by the AutoNMT backend.

::: autonmt.core.data.translation_dataset

## Samplers

::: autonmt.core.samplers
