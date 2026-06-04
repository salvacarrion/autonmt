# Backends

The shared translator contract, its config objects, and the three concrete engines.
See [Choosing a backend](../guide/backends/choosing.md) for the narrative version.

## BaseTranslator

::: autonmt.backends._base.translation_engine

## Configuration

::: autonmt.backends._base.config

## AutonmtTranslator

::: autonmt.backends.autonmt.translation_engine.AutonmtTranslator

## LMTrainer

The trainer for decoder-only language models (`fit` / `evaluate` / `generate`). See
[Train a language model](../guide/data/lm-corpora.md) and
[Text generation & sampling](../guide/translation/text-generation.md).

::: autonmt.backends.lm.trainer.LMTrainer

## MLMTrainer

The trainer for encoder-only masked language models (`fit` / `evaluate` / `fill_mask`). See
[Pretrain a masked LM](../how-to/pretrain-masked-lm.md).

::: autonmt.backends.lm.mlm_trainer.MLMTrainer

## HuggingFaceTranslator

::: autonmt.backends.huggingface.translation_engine.HuggingFaceTranslator

## FairseqTranslator

::: autonmt.backends.fairseq.translation_engine.FairseqTranslator
