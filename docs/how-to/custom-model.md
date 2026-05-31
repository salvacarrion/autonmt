# Add a custom model

A new architecture is a subclass of the seq2seq base that fills in **three forward methods**
and returns logits shaped `(batch, length, vocab)`. The `states` you return from
`forward_encoder` is threaded through `forward_decoder` by every
[decoder](../guide/translation/decoding.md), so any architecture works with greedy, beam, and
sampling search unchanged. (The full contract is in
[Writing your own model](../guide/models/custom-models.md).)

```python
import torch.nn as nn
from autonmt.core.nn.seq2seq import LitSeq2Seq

class TiedTransformer(LitSeq2Seq):
    def __init__(self, src_vocab_size, tgt_vocab_size, padding_idx, d_model=256, **kw):
        super().__init__(src_vocab_size, tgt_vocab_size, padding_idx,
                         architecture="tied-transformer", **kw)
        # ... build encoder/decoder, embeddings, output projection ...

    def forward_encoder(self, x, x_len, **kw):
        ...
        return None, states                       # states: whatever the decoder needs

    def forward_decoder(self, y, y_len, states, **kw):
        ...
        return logits, states                     # logits: (B, L, V)

    def forward_enc_dec(self, x, x_len, y, y_len, **kw):
        _, states = self.forward_encoder(x, x_len)
        logits, _ = self.forward_decoder(y, y_len, states)
        return logits
```

Then it plugs straight into the translator — `from_vocabs` works for your subclass exactly
as for the built-ins:

```python
trainer = AutonmtTranslator.from_dataset(
    train_ds, model=TiedTransformer.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab, run_prefix="tied")
```

!!! tip "Reuse the building blocks"
    Before writing layers from scratch, check
    [`autonmt.core.nn.layers`](../guide/models/building-blocks.md) — `RMSNorm`, `SwiGLU`,
    `RotaryPositionalEmbedding`, and the KV-cache-aware `IncrementalTransformerDecoder` are
    ready to drop in. Set `supports_incremental_decoding = True` on your model if your decoder
    supports KV-cached single-token steps, to get fast beam search.
