# Core

AutoNMT's in-house neural engine: the Lightning base class, built-in architectures, the layer
library, decoding strategies, samplers, and the torch dataset. For the narrative version, see
[Models](../guide/models/using-a-model.md).

## LitSeq2Seq

The base every custom model inherits. See [Writing your own
model](../guide/models/custom-models.md).

::: autonmt.core.nn.seq2seq.LitSeq2Seq

## Built-in models

::: autonmt.core.nn.models

## Layers

The building blocks for custom architectures (positional encodings, RMSNorm, SwiGLU, the
KV-cache-aware Transformer decoder).

::: autonmt.core.nn.layers

## Decoding strategies

See [Decoding strategies](../guide/translation/decoding.md) for the intuition and math.

::: autonmt.core.decoding

## Translation dataset

The torch `Dataset` used internally by the AutoNMT backend. See [Bucketing &
batching](../guide/training/bucketing.md).

::: autonmt.core.data.translation_dataset

## Samplers

::: autonmt.core.samplers
