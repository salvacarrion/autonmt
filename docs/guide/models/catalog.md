# Model catalog

The native engine ships seven encoder–decoder architectures, all under
`autonmt.core.nn.models` and all subclasses of the same seq2seq base, so they share the
[`from_vocabs`](using-a-model.md) constructor and the whole training loop. Pick by the
research question you're asking.

| Class | Family | Reach for it when… |
| --- | --- | --- |
| `Transformer` | self-attention enc–dec | the default; almost always your starting point |
| `SimpleRNN` | RNN enc–dec | you want a plain recurrent baseline (no attention) |
| `ContextRNN` | RNN enc–dec | you want the encoder context injected at every step |
| `BahdanauRNN` | RNN + additive attention | classic attention baseline (Bahdanau et al.) |
| `LuongRNN` | RNN + multiplicative attention | the other classic attention baseline (Luong et al.) |
| `ConvS2S` | convolutional enc–dec | a fully-convolutional, attention-free-ish baseline |
| `MLP` | feed-forward | a tiny non-recurrent toy/baseline, handy for tests |

## Transformer

The default (Vaswani et al., 2017). The constructor exposes the standard knobs; defaults
build a small model that trains quickly:

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
  positional encodings are added, so the two have comparable magnitude (paper §3.4).
- **`tie_embeddings`.** Shares the decoder input embedding with the output projection (Press
  & Wolf, 2017) — fewer parameters, standard in NMT. Requires a compatible (often
  [merged](../data/vocabularies.md#separate-vs-shared-merged-vocabularies)) vocabulary.
- **`norm_first` (Pre-LN vs Post-LN).** Pre-LN puts LayerNorm before each sub-block and
  tends to train more stably without careful warmup; Post-LN is the original formulation.
- **KV-cached decoding.** `supports_incremental_decoding = True`, so the decoders feed only
  the last token each step and reuse cached keys/values — turning the per-step cost from
  $O(L^2)$ to $O(L)$. Transparent to you; it just makes beam search fast.

!!! info "Positional encodings, briefly"
    Attention is order-agnostic — without help, a Transformer sees a *set* of tokens, not a
    sequence. **Positional embeddings** add per-position information so word order matters.
    `learned=False` uses **sinusoidal** (fixed, generalizes to longer sequences); `True`
    uses **learned** (trainable). A **rotary** variant is also available — see
    [Building blocks](building-blocks.md).

## The RNN family

All four share a recurrent core and the same constructor knobs (`encoder_hidden_dim`,
`encoder_n_layers`, `encoder_bidirectional`, `teacher_force_ratio`, …), and pick the cell
with `base_rnn="rnn" | "lstm" | "gru"`:

- **`SimpleRNN`** (Sutskever et al., 2014) — the encoder compresses the source into a final
  hidden state; the decoder is seeded with it and generates token by token. No attention:
  the decoder sees the source *only* through that fixed-size state.
- **`ContextRNN`** — like `SimpleRNN`, but the encoder context is **injected at every decode
  step**, not just used as the initial state, so it doesn't have to survive in the hidden
  state alone.
- **`BahdanauRNN`** (Bahdanau et al., 2015) — adds **additive attention**: at each step the
  decoder computes a weighted read over *all* encoder states, learning where to look.
- **`LuongRNN`** (Luong et al., 2015) — the same idea with **multiplicative (dot-product)
  attention**, the other canonical formulation.

!!! info "Why attention mattered (and still does)"
    A plain RNN forces the entire source meaning through one fixed-size vector — a
    bottleneck that hurts long sentences. **Attention** lets the decoder look back at every
    source position with learned weights at each step, removing the bottleneck. It's the
    idea the Transformer later took to its logical extreme (attention *only*, no recurrence).

!!! note "RNNs and bucketing"
    Recurrent models benefit from packed sequences (`packed_sequence=True`), which AutoNMT
    only allows together with length [bucketing](../training/bucketing.md). The catalog
    classes set this up for you when you opt in.

## ConvS2S

A fully convolutional encoder–decoder (Gehring et al., 2017): stacked convolutions with
gated linear units replace recurrence, so the source is processed in parallel. A useful
non-recurrent, non-self-attention point of comparison.

## MLP

A minimal feed-forward seq2seq with no recurrence or attention. It exists as a tiny,
fast baseline and as a fixture for tests — not a serious translation model, but handy when
you want to exercise the *pipeline* without waiting on a real architecture.

---

Building a variant of one of these, or something new? See the reusable
[Building blocks](building-blocks.md) and [Writing your own model](custom-models.md). Full
signatures live in the [API reference](../../reference/core.md).
