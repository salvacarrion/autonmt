# Custom models

AutoNMT ships several built-in architectures (`Transformer`, `ConvS2S`, the RNN family,
`MLP`) under [`autonmt.core.nn.models`](../reference/core.md). When you want your own, you
subclass one base class and plug it in exactly like a built-in - this is the
[extension-first](../concepts/philosophy.md#4-minimal-core-extend-at-the-edges) design in
action.

## The contract

Inherit from [`LitSeq2Seq`][autonmt.core.nn.seq2seq.LitSeq2Seq] (a `LightningModule`) and
implement three methods. Logits must come out shaped **`(batch, length, vocab)`**.

```python
from autonmt.core.nn.seq2seq import LitSeq2Seq

class MyModel(LitSeq2Seq):
    def __init__(self, src_vocab_size, tgt_vocab_size, padding_idx, **kwargs):
        super().__init__(src_vocab_size, tgt_vocab_size, padding_idx, **kwargs)
        # build your layers here

    def forward_encoder(self, x, x_len, **kwargs):
        # returns encoder states
        ...

    def forward_decoder(self, y, y_len, states, **kwargs):
        # one decode step / pass given encoder states → logits
        ...

    def forward_enc_dec(self, x, x_len, y, y_len, **kwargs):
        # full teacher-forced forward → logits (batch, length, vocab)
        ...
```

`LitSeq2Seq` already implements the Lightning plumbing - training/validation steps, the
optimizer, loss, and the decoding entry points - so you only write the architecture.

## Plug it in

The model is constructed from the vocabularies, exactly like the built-in `Transformer`:

```python
src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)

trainer = AutonmtTranslator.from_dataset(
    train_ds,
    model=MyModel.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    run_prefix="mymodel",
)
trainer.fit(train_ds, config=FitConfig(max_epochs=10, batch_size=128))
```

`from_vocabs` is a classmethod on the base - it reads `src_vocab_size`, `tgt_vocab_size`,
and `padding_idx` off the vocabularies and forwards any extra kwargs to your `__init__`, so
you can pass hyperparameters straight through:

```python
MyModel.from_vocabs(src_vocab, tgt_vocab, hidden_dim=512, num_layers=6, dropout=0.1)
```

## Decoding strategies

Decoding is decoupled from the model. The strategies live under
[`autonmt.core.decoding`](../reference/core.md) - `GreedySearch`, `BeamSearch`,
`MultinomialSampling`, `TopkSampling`, `ToppSampling` - built on the `BaseSearch` contract.
Pick one at predict time:

```python
from autonmt.core.decoding import BeamSearch
# wire your chosen search into the translator / predict config
```

Because the search algorithms are separate from the architecture, the same custom model
works with greedy, beam, and sampling decoders without changes.

## Tips

- Keep architecture knobs as `__init__` kwargs and pass them via `from_vocabs(...)` - they
  flow into the run config dump for [reproducibility](../concepts/reproducibility.md).
- Start by reading
  [`autonmt/core/nn/models/transformer/transformer.py`](https://github.com/salvacarrion/autonmt/blob/main/autonmt/core/nn/models/transformer/transformer.py)
  as a reference implementation of all three `forward_*` methods.
- Reusable building blocks (positional encodings, RMSNorm, SwiGLU, an incremental decoder)
  live under [`autonmt.core.nn.layers`](https://github.com/salvacarrion/autonmt/tree/main/autonmt/core/nn/layers).
