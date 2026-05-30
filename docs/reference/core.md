# Models & decoding

AutoNMT's in-house neural engine: the Lightning base class, built-in architectures, the layer
library, decoding strategies, samplers, and the torch dataset. For the narrative version, see
[The AutoNMT toolkit](../toolkit/overview.md).

## LitSeq2Seq

The base every custom model inherits. See [Models & the `LitSeq2Seq`
contract](../toolkit/models.md).

::: autonmt.core.nn.seq2seq.LitSeq2Seq

## Built-in models

::: autonmt.core.nn.models

## Layers

The building blocks for custom architectures (positional encodings, RMSNorm, SwiGLU, the
KV-cache-aware Transformer decoder).

::: autonmt.core.nn.layers

## Decoding strategies

See [Decoding strategies](../toolkit/decoding.md) for the intuition and math.

::: autonmt.core.decoding

## Translation dataset

The torch `Dataset` used internally by the AutoNMT backend. See [Samplers & the
TranslationDataset](../toolkit/data-pipeline.md).

::: autonmt.core.data.translation_dataset

## Samplers

::: autonmt.core.samplers
