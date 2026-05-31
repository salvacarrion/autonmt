# Building blocks

When you build or modify an architecture, you rarely start from `nn.Linear`. AutoNMT's
`autonmt.core.nn.layers` package collects the modern components the built-in models are
made of, so a [custom model](custom-models.md) composes them instead of reimplementing
them. All are importable from `autonmt.core.nn.layers`.

| Building block | What it is |
| --- | --- |
| `PositionalEmbedding` | dispatcher: picks sinusoidal or learned absolute positions |
| `SinusoidalPositionalEmbedding` | fixed sinusoidal positions (generalizes to longer sequences) |
| `LearnedPositionalEmbedding` | trainable absolute positions |
| `RotaryPositionalEmbedding` | rotary positions (RoPE) — applied inside attention |
| `RMSNorm` | RMS normalization, a lighter LayerNorm variant |
| `SwiGLU` | gated feed-forward activation |
| `IncrementalTransformerDecoder` / `IncrementalTransformerDecoderLayer` | KV-cache-aware decoder |

## Positional encodings

!!! info "Why positions need encoding"
    Self-attention is permutation-invariant: with no extra signal, a Transformer sees a
    *bag* of tokens, not an ordered sequence. Positional encodings inject "where am I" so
    word order carries meaning.

- **Sinusoidal** (`SinusoidalPositionalEmbedding`) — fixed sine/cosine patterns; no
  parameters, and extrapolates to sequences longer than seen in training.
- **Learned** (`LearnedPositionalEmbedding`) — a trainable embedding per position; flexible
  but capped at `max_positions`.
- **Rotary / RoPE** (`RotaryPositionalEmbedding`) — instead of *adding* a position vector,
  it *rotates* the query/key vectors by a position-dependent angle inside attention, which
  encodes *relative* position and tends to generalize well to longer contexts.

`PositionalEmbedding` is the dispatcher the `Transformer` uses (`learned=` flips between
sinusoidal and learned); `pos_embedding_at(...)` is a small helper for fetching the encoding
at a given step during incremental decoding.

## Normalization & feed-forward

- **`RMSNorm`** — normalizes by the root-mean-square of activations (no mean subtraction,
  no bias). Cheaper than LayerNorm and common in recent architectures.
- **`SwiGLU`** — a gated feed-forward block (a SiLU-gated linear unit) that often
  outperforms a plain ReLU/GELU MLP at equal parameter budget; a drop-in for the
  position-wise FFN.

## The incremental (autoregressive) decoder

This is the "iterative" counterpart to the parallel Transformer decoder, and it's worth a
moment because it's where training and inference genuinely differ.

!!! info "Parallel training vs iterative decoding"
    During **training**, the whole target sentence is known, so the decoder runs **once**
    over all positions in parallel (with a causal mask). During **inference**, the target is
    produced **one token at a time** — each new token depends on the ones already generated.
    A naïve loop would re-encode the entire prefix at every step: $O(L)$ steps each costing
    $O(L)$, i.e. $O(L^2)$ work. The **incremental decoder** instead **caches** the keys and
    values computed for previous positions, so each step only processes the *new* token —
    bringing the per-step cost down to $O(L)$. This is what makes beam search affordable.

`IncrementalTransformerDecoder` / `IncrementalTransformerDecoderLayer` implement that
KV-cache-aware decoding. Two practical properties:

- **Parameter-compatible** with PyTorch's `nn.TransformerDecoder` — same weight layout, so a
  checkpoint trained with the standard module loads into the incremental one unchanged. You
  get fast decoding without retraining.
- **Driven by `incremental_state`** — the [search algorithms](../translation/decoding.md)
  pass an `incremental_state={}` dict that the decoder fills and threads through steps; a
  model advertises support via `supports_incremental_decoding = True`. The `Transformer` and
  the RNN family both set it.

---

With the pieces in hand, the next page assembles them into a new architecture:
[Writing your own model](custom-models.md).
