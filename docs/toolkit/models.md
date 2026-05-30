# Models & the `LitSeq2Seq` contract

Every model AutoNMT trains is a subclass of
[`LitSeq2Seq`](../reference/core.md#autonmt.core.nn.seq2seq.LitSeq2Seq) — a PyTorch Lightning
base that standardizes the training/validation loop, optimizer/scheduler configuration, and
the encoder–decoder interface the [decoders](decoding.md) rely on. You can use a built-in
architecture or implement your own by filling in three methods.

## The contract

A `LitSeq2Seq` is built from vocab sizes and the pad id, and must implement three forward
methods:

```python
class LitSeq2Seq(pl.LightningModule):
    def __init__(self, src_vocab_size, tgt_vocab_size, padding_idx,
                 packed_sequence=False, architecture=None, **kwargs): ...

    # The three methods every architecture implements:
    def forward_encoder(self, x, x_len, **kwargs): ...
    def forward_decoder(self, y, y_len, states, **kwargs): ...
    def forward_enc_dec(self, x, x_len, y, y_len, **kwargs): ...
```

| Method | Called during | Returns |
| --- | --- | --- |
| `forward_encoder(x, x_len)` | encoding the source once | `(_, states)` — the encoder memory / hidden state |
| `forward_decoder(y, y_len, states)` | each decode step / the full target | `(logits, states)`, logits shaped `(batch, length, vocab)` |
| `forward_enc_dec(x, x_len, y, y_len)` | training (teacher forcing) | `logits` shaped `(batch, length, vocab)` |

The key design point is the **`states` hand-off**: `forward_encoder` produces whatever the
decoder needs to remember about the source (Transformer encoder memory + padding mask; RNN
hidden state), and the [decoder](decoding.md) threads that object through every step. This is
what lets one generic search loop drive every architecture — the search code never needs to
know whether it's running a Transformer or an RNN.

!!! note "Logits, not probabilities"
    `forward_*` returns raw **logits** of shape `(batch, length, vocab)`. The training loss
    applies the softmax/cross-entropy; the decoders apply `log_softmax` themselves. Returning
    logits keeps the numerics in one place and lets samplers apply a temperature before
    normalizing.

### Building from vocabularies

`from_vocabs` is the convenience constructor — it infers sizes and the pad id from the
vocabularies, so you don't repeat them:

```python
from autonmt.core.nn.models import Transformer

model = Transformer.from_vocabs(src_vocab, tgt_vocab)        # sizes + pad id inferred
model = Transformer.from_vocabs(src_vocab, tgt_vocab, encoder_layers=6, dropout=0.3)  # override anything
```

It asserts `src_vocab.pad_id == tgt_vocab.pad_id` (true by default); pass `padding_idx=`
explicitly if you build vocabularies with different pad ids.

## Built-in architectures

All live under `autonmt.core.nn.models`:

| Class | Family | Notes |
| --- | --- | --- |
| `Transformer` | self-attention enc-dec | The default. KV-cached incremental decoding. |
| `SimpleRNN` | RNN enc-dec | Plain recurrent baseline |
| `ContextRNN` | RNN enc-dec | Injects the encoder context at each step |
| `BahdanauRNN` | RNN + attention | Additive ("Bahdanau") attention |
| `LuongRNN` | RNN + attention | Multiplicative ("Luong") attention |
| `ConvS2S` | convolutional enc-dec | Fully convolutional seq2seq |
| `MLP` | feed-forward | Tiny non-recurrent baseline, useful for tests |

### The Transformer

The default architecture (Vaswani et al., 2017). Its constructor exposes the standard knobs;
the defaults build a small model that trains quickly:

```python
Transformer(
    src_vocab_size, tgt_vocab_size,
    encoder_embed_dim=256, decoder_embed_dim=256,
    encoder_layers=3, decoder_layers=3,
    encoder_attention_heads=8, decoder_attention_heads=8,
    encoder_ffn_embed_dim=512, decoder_ffn_embed_dim=512,
    dropout=0.1, activation_fn="relu",
    max_src_positions=1024, max_tgt_positions=1024,
    learned=False,          # learned vs sinusoidal positional embeddings
    tie_embeddings=False,   # share target input embedding with the output projection
    norm_first=False,       # Pre-LN (True) vs Post-LN (False)
)
```

A few choices worth understanding:

- **Embedding scaling.** Token embeddings are scaled by $\sqrt{d_{\text{model}}}$ before
  positional encodings are added, so the two have comparable magnitude (per the original
  paper).
- **`tie_embeddings`.** Shares the decoder input embedding with the output projection (Press &
  Wolf, 2017) — fewer parameters, standard in NMT. Requires a compatible (often
  [merged](../data/vocabularies.md#separate-vs-shared-merged-vocabularies)) vocabulary.
- **`norm_first` (Pre-LN vs Post-LN).** Pre-LN puts LayerNorm before each sub-block and tends
  to train more stably without careful warmup; Post-LN is the original formulation.
- **KV-cached decoding.** The Transformer sets `supports_incremental_decoding = True`, so the
  decoders feed only the last token each step and reuse cached keys/values — turning the
  per-step cost from $O(L^2)$ to $O(L)$. This is transparent to you; it just makes beam search
  fast.

!!! info "Positional encodings, briefly"
    Attention is order-agnostic — without help, a Transformer sees a *set* of tokens, not a
    sequence. **Positional embeddings** add per-position information so word order matters.
    AutoNMT's `PositionalEmbedding` picks **sinusoidal** (fixed, generalizes to longer
    sequences) or **learned** (trainable) via `learned=`. A **rotary** variant
    (`RotaryPositionalEmbedding`) is also available in the layer library.

## The layer library

If you build custom architectures, `autonmt.core.nn.layers` gives you modern building blocks:

| Layer | What it is |
| --- | --- |
| `SinusoidalPositionalEmbedding` / `LearnedPositionalEmbedding` | absolute position encodings (dispatched by `PositionalEmbedding`) |
| `RotaryPositionalEmbedding` | rotary position encoding (RoPE) |
| `RMSNorm` | RMSNorm normalization (a lighter LayerNorm variant) |
| `SwiGLU` | gated feed-forward activation |
| `IncrementalTransformerDecoder` / `IncrementalTransformerDecoderLayer` | KV-cache-aware decoder (drop-in for `nn.TransformerDecoder`, identical parameter layout) |

The incremental decoder is parameter-compatible with PyTorch's built-in, so checkpoints load
unchanged — you get fast incremental decoding without retraining.

## Training-time behavior you inherit

Subclassing `LitSeq2Seq` gives you, for free:

- **`configure_optimizers`** — resolves the optimizer string (`"adam"`, `"adamw"`, `"sgd"`, …)
  and builds the LR scheduler (`"noam"` / `"inverse_sqrt"` / callable / instance) with
  per-step updates. Configured from [`FitConfig`](training.md#fitconfig).
- **`count_parameters()`** — total / trainable / non-trainable counts (also reported in the
  run metadata).
- The standard **training/validation steps** with teacher forcing, cross-entropy (ignoring
  `<pad>`), and optional decoded-sample logging (`print_samples`).

## Writing your own model

The minimal recipe — implement the three methods, return logits `(batch, length, vocab)`:

```python
from autonmt.core.nn.seq2seq import LitSeq2Seq

class MyModel(LitSeq2Seq):
    def __init__(self, src_vocab_size, tgt_vocab_size, padding_idx, **kw):
        super().__init__(src_vocab_size, tgt_vocab_size, padding_idx,
                         architecture="my-model", **kw)
        # ... build your encoder / decoder ...

    def forward_encoder(self, x, x_len, **kw):
        ...
        return None, states                     # states: whatever the decoder needs

    def forward_decoder(self, y, y_len, states, **kw):
        ...
        return logits, states                   # logits: (B, L, V)

    def forward_enc_dec(self, x, x_len, y, y_len, **kw):
        _, states = self.forward_encoder(x, x_len)
        logits, _ = self.forward_decoder(y, y_len, states)
        return logits
```

Then it plugs straight into the translator: `AutonmtTranslator(model=MyModel.from_vocabs(...),
...)`. A worked example, including how to expose `states` correctly for beam search, is in
[Extending AutoNMT](../extending/index.md#a-custom-model).

---

Next: how those logits become translations — **[Decoding strategies](decoding.md)**.
