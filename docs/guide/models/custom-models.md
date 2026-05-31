# Writing your own model

This is the point of AutoNMT's native engine: when your research *is* the architecture, you
write the model and inherit everything else — the training loop, optimizer/scheduler
plumbing, checkpointing, decoding, scoring, and reporting. A new architecture is a subclass
that fills in **three methods** and returns logits; nothing else changes.

## The encoder–decoder interface

Models subclass the seq2seq base (`LitSeq2Seq`, a PyTorch Lightning module). It's built from
the vocab sizes and the pad id, and asks you to implement three forward methods:

```python
class LitSeq2Seq(pl.LightningModule):
    def __init__(self, src_vocab_size, tgt_vocab_size, padding_idx,
                 packed_sequence=False, architecture=None, **kwargs): ...

    def forward_encoder(self, x, x_len, **kwargs): ...
    def forward_decoder(self, y, y_len, states, **kwargs): ...
    def forward_enc_dec(self, x, x_len, y, y_len, **kwargs): ...
```

| Method | Called during | Returns |
| --- | --- | --- |
| `forward_encoder(x, x_len)` | encoding the source once | `(_, states)` — the encoder memory / hidden state |
| `forward_decoder(y, y_len, states)` | each decode step / the full target | `(logits, states)`, logits `(batch, length, vocab)` |
| `forward_enc_dec(x, x_len, y, y_len)` | training (teacher forcing) | `logits` shaped `(batch, length, vocab)` |

The key design point is the **`states` hand-off**: `forward_encoder` produces whatever the
decoder needs to remember about the source (Transformer encoder memory + padding mask; an
RNN hidden state), and the [decoder](../translation/decoding.md) threads that object through
every step. That's what lets *one* generic search loop drive every architecture — the
search code never needs to know whether it's running a Transformer or an RNN.

!!! note "Logits, not probabilities"
    `forward_*` returns raw **logits** of shape `(batch, length, vocab)`. The training loss
    applies softmax/cross-entropy; the decoders apply `log_softmax` themselves. Returning
    logits keeps the numerics in one place and lets samplers apply a temperature before
    normalizing.

## The minimal recipe

Implement the three methods, compose [building blocks](building-blocks.md) for the internals,
and call `super().__init__` with a descriptive `architecture` tag (it shows up in run
metadata and reports):

```python
from autonmt.core.nn.seq2seq import LitSeq2Seq

class MyModel(LitSeq2Seq):
    def __init__(self, src_vocab_size, tgt_vocab_size, padding_idx, **kw):
        super().__init__(src_vocab_size, tgt_vocab_size, padding_idx,
                         architecture="my-model", **kw)
        # ... build your encoder / decoder from autonmt.core.nn.layers ...

    def forward_encoder(self, x, x_len, **kw):
        ...
        return None, states                  # states: whatever the decoder needs

    def forward_decoder(self, y, y_len, states, **kw):
        ...
        return logits, states                # logits: (B, L, V)

    def forward_enc_dec(self, x, x_len, y, y_len, **kw):
        _, states = self.forward_encoder(x, x_len)
        logits, _ = self.forward_decoder(y, y_len, states)
        return logits
```

Then it plugs straight into a translator — `from_vocabs` works for your subclass exactly as
it does for the built-ins:

```python
trainer = AutonmtTranslator.from_dataset(
    train_ds, model=MyModel.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab, run_prefix="my-model",
)
```

## Supporting fast decoding (optional)

By default your model decodes correctly but recomputes the prefix each step. To opt into
[KV-cached incremental decoding](building-blocks.md#the-incremental-autoregressive-decoder),
set `supports_incremental_decoding = True` and have `forward_decoder` honor the
`incremental_state` dict the search loop passes in (process only the last token, cache the
rest). The built-in `Transformer` and RNNs are worked references.

---

For a complete, runnable custom-model example — including how to expose `states` correctly
for beam search — see [How-to → Add a custom model](../../how-to/custom-model.md). For the
exact base-class signatures, the [API reference](../../reference/core.md).
