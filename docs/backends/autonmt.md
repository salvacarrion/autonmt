# AutoNMT backend (PyTorch Lightning)

[`AutonmtTranslator`][autonmt.backends.autonmt.translation_engine.AutonmtTranslator] is the
default backend. It wraps a [`LitSeq2Seq`](../guides/custom-models.md) model in PyTorch
Lightning and owns the DataLoaders, callbacks (EarlyStopping, ModelCheckpoint), and loggers
(TensorBoard / Weights & Biases).

## Constructing it

`from_dataset` is the idiomatic constructor - it resolves the run directory and run name
from the dataset cell, so checkpoints and logs land in the right place automatically:

```python
from autonmt.backends import AutonmtTranslator
from autonmt.core.nn.models import Transformer

src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)

trainer = AutonmtTranslator.from_dataset(
    train_ds,
    model=Transformer.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    run_prefix="exp",
)
```

The `run_prefix` becomes part of the run name under
`models/autonmt/runs/<run>/` - see [On-disk layout](../concepts/on-disk-layout.md).

## Training

```python
from autonmt.backends._base.config import FitConfig

trainer.fit(train_ds, config=FitConfig(
    max_epochs=10, batch_size=128, learning_rate=1e-3,
    weight_decay=0.0, gradient_clip_val=1.0,
    patience=5,           # EarlyStopping
    seed=42,              # seeds Python/NumPy/Torch/Lightning
))
```

Common `FitConfig` knobs map onto Lightning's `Trainer` and the model's optimizer. Anything
Lightning-specific that AutoNMT doesn't model (e.g. `strategy`, `precision`) passes through
`**kwargs` untouched.

### Weights & Biases

Install the extra (`pip install -e '.[wandb]'`) and pass `wandb_params`:

```python
trainer.fit(train_ds, config=FitConfig(max_epochs=10),
            wandb_params={"project": "autonmt", "name": "exp-1"})
```

TensorBoard logging is on by default - point `tensorboard --logdir` at the run's `logs/`.

## Prediction & scoring

```python
from autonmt.backends._base.config import PredictConfig

scores = trainer.predict(test_datasets, config=PredictConfig(
    metrics={"bleu", "chrf"},
    beams=[1, 5],                 # greedy + beam-5, each its own report column
    load_checkpoint="best",       # or a specific checkpoint path
    preprocess_fn=preprocess_predict,
    eval_mode="compatible",
))
```

`predict()` encodes each eval split with the run's SentencePiece model, decodes the
hypotheses back to text, scores them, and returns the per-run dict that feeds
[`generate_report`](../guides/reports.md).

## Built-in models

Importable from [`autonmt.core.nn.models`](../reference/core.md):

| Model                               | Class                                                |
| ----------------------------------- | ---------------------------------------------------- |
| Transformer                         | `Transformer`                                        |
| Convolutional seq2seq               | `ConvS2S`                                            |
| RNN (vanilla / context / attention) | `SimpleRNN`, `ContextRNN`, `BahdanauRNN`, `LuongRNN` |
| MLP (baseline)                      | `MLP`                                                |

Bring your own by subclassing `LitSeq2Seq` - see [Custom models](../guides/custom-models.md).

## API reference

See [`AutonmtTranslator`][autonmt.backends.autonmt.translation_engine.AutonmtTranslator] in
the [backends API reference](../reference/backends.md#autonmttranslator).
