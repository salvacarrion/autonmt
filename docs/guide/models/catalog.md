# Model catalog

The native engine ships seven encoder–decoder architectures, all under
`autonmt.core.nn.models` and all subclasses of the same seq2seq base, so they share the
[`from_vocabs`](using-a-model.md) constructor and the whole training loop. Pick by the
research question you're asking. It also ships a decoder-only **[`GPT`](#gpt)** for
[language modelling](../data/lm-corpora.md) — same engine, a different base (it's trained with
[`LMTrainer`](../../reference/backends.md) rather than `AutonmtTranslator`).

| Class         | Family                         | Reach for it when…                                  |
| ------------- | ------------------------------ | --------------------------------------------------- |
| `Transformer` | self-attention enc–dec         | the default; almost always your starting point      |
| `SimpleRNN`   | RNN enc–dec                    | you want a plain recurrent baseline (no attention)  |
| `ContextRNN`  | RNN enc–dec                    | you want the encoder context injected at every step |
| `BahdanauRNN` | RNN + additive attention       | classic attention baseline (Bahdanau et al.)        |
| `LuongRNN`    | RNN + multiplicative attention | the other classic attention baseline (Luong et al.) |
| `ConvS2S`     | convolutional enc–dec          | a fully-convolutional, attention-free-ish baseline  |
| `MLP`         | feed-forward                   | a tiny non-recurrent toy/baseline, handy for tests  |

## Transformer

The default ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). The constructor exposes the standard knobs; defaults
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
- **`tie_embeddings`.** Shares the decoder input embedding with the output projection ([Press & Wolf, 2017](https://arxiv.org/abs/1608.05859)) — fewer parameters, standard in NMT. Requires a compatible (often
  [merged](../data/vocabularies.md#separate-vs-shared-merged-vocabularies)) vocabulary.
- **`norm_first` (Pre-LN vs Post-LN).** Pre-LN puts
  [LayerNorm](https://arxiv.org/abs/1607.06450) before each sub-block and tends to train more
  stably without careful warmup ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745));
  Post-LN is the original formulation.
- **KV-cached decoding.** `supports_incremental_decoding = True`, so the decoders feed only
  the last token each step and reuse cached keys/values — turning the per-step cost from
  $O(L^2)$ to $O(L)$. Transparent to you; it just makes beam search fast.

!!! info "Positional encodings, briefly"
Attention is order-agnostic — without help, a Transformer sees a _set_ of tokens, not a
sequence. **Positional embeddings** add per-position information so word order matters.
`learned=False` uses **sinusoidal** (fixed, generalizes to longer sequences); `True`
uses **learned** (trainable). A **rotary** variant is also available — see
[Building blocks](building-blocks.md).

## The RNN family

All four share a recurrent core and the same constructor knobs (`encoder_hidden_dim`,
`encoder_n_layers`, `encoder_bidirectional`, `teacher_force_ratio`, …), and pick the cell
with `base_rnn="rnn" | "lstm" | "gru"`:

- **`SimpleRNN`** ([Sutskever et al., 2014](https://arxiv.org/abs/1409.3215)) — the encoder compresses the source into a final
  hidden state; the decoder is seeded with it and generates token by token. No attention:
  the decoder sees the source _only_ through that fixed-size state.
- **`ContextRNN`** ([Cho et al., 2014](https://arxiv.org/abs/1406.1078)) — like `SimpleRNN`,
  but the encoder context is **injected at every decode step**, not just used as the initial
  state, so it doesn't have to survive in the hidden state alone. (This is the RNN
  Encoder–Decoder that also introduced the GRU.)
- **`BahdanauRNN`** ([Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473)) — adds **additive attention**: at each step the
  decoder computes a weighted read over _all_ encoder states, learning where to look.
- **`LuongRNN`** ([Luong et al., 2015](https://arxiv.org/abs/1508.04025)) — the same idea with **multiplicative (dot-product)
  attention**, the other canonical formulation.

!!! info "Why attention mattered (and still does)"
A plain RNN forces the entire source meaning through one fixed-size vector — a
bottleneck that hurts long sentences. **Attention** lets the decoder look back at every
source position with learned weights at each step, removing the bottleneck. It's the
idea the Transformer later took to its logical extreme (attention _only_, no recurrence).

!!! note "RNNs and bucketing"
Recurrent models benefit from packed sequences (`packed_sequence=True`), which AutoNMT
only allows together with length [bucketing](../training/bucketing.md). The catalog
classes set this up for you when you opt in.

## ConvS2S

A fully convolutional encoder–decoder ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)): stacked convolutions with
gated linear units replace recurrence, so the source is processed in parallel. A useful
non-recurrent, non-self-attention point of comparison.

## MLP

A minimal feed-forward seq2seq with no recurrence or attention. It exists as a tiny,
fast baseline and as a fixture for tests — not a serious translation model, but handy when
you want to exercise the _pipeline_ without waiting on a real architecture.

## GPT

A nanoGPT-style **decoder-only** Transformer for language modelling
([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)).
Unlike the models above, it has no encoder and no cross-attention — a single causal stack
predicts the next token — so it subclasses the LM base (`LitLM`, not the seq2seq base) and is
trained with [`LMTrainer`](../../reference/backends.md). Defaults follow the modern
([LLaMA, Touvron et al., 2023](https://arxiv.org/abs/2302.13971)) recipe:

```python
GPT(
    vocab_size, padding_idx=None,
    embed_dim=256, num_layers=4, num_heads=8,
    ffn_dim=None,            # SwiGLU hidden dim; defaults to ~2/3·4·embed_dim
    dropout=0.1,
    max_seq_len=1024,        # longest context (bounds positions); block_size must be ≤ this
    use_rope=True,           # rotary positions (else learned absolute)
    tie_embeddings=True,     # share token embedding with the output projection
    norm_eps=1e-6,
)
```

- **Modern stack.** [Rotary positions (RoPE)](building-blocks.md#positional-encodings),
  [RMSNorm](building-blocks.md#normalization-feed-forward), and
  [SwiGLU](building-blocks.md#normalization-feed-forward) feed-forwards in a pre-norm block —
  the same [building blocks](building-blocks.md) the catalog shares, assembled around the new
  [`CausalSelfAttention`](building-blocks.md#causal-self-attention).
- **Weight tying.** `tie_embeddings=True` shares the input embedding with the output
  projection ([Press & Wolf, 2017](https://arxiv.org/abs/1608.05859)).
- **KV-cached decoding.** `supports_incremental_decoding = True`, so
  [generation](../translation/text-generation.md) feeds one token per step and reuses cached
  keys/values — $O(L)$ per step instead of $O(L^2)$.

Build it sized to a corpus with `GPT.from_corpus(corpus, ...)`, which reads the vocabulary size
and pad id from the corpus's tokenizer. See [Train a language model](../../how-to/train-language-model.md).

## MLMTransformer

A BERT-style **encoder-only** masked language model
([Devlin et al., 2019](https://arxiv.org/abs/1810.04805)). Where `GPT` is causal and
generates, this is **bidirectional** — every position attends to the whole sequence — and
predicts randomly _masked_ tokens. It reuses PyTorch's `nn.TransformerEncoder` (the same
bidirectional stack the built-in `Transformer`'s encoder uses), so the only thing new on the
encoder-only path is the objective, not the attention. Like `GPT` it subclasses an LM base
(`LitMLM`) and trains with a dedicated trainer ([`MLMTrainer`](../../reference/backends.md)).

```python
MLMTransformer(
    vocab_size, padding_idx=None,
    embed_dim=256, num_layers=4, num_heads=8,
    ffn_dim=None,            # defaults to 4 * embed_dim
    dropout=0.1,
    max_seq_len=1024,        # block_size must be ≤ this
    activation="gelu",       # BERT's FFN activation
    tie_embeddings=True,     # share token embedding with the MLM head
    norm_first=False,        # Pre-LN vs BERT's Post-LN
)
```

- **No generation.** An MLM doesn't decode left to right; instead of `generate` you call
  `MLMTrainer.fill_mask("the <mask> fox …")` to predict the masked positions.
- **Reserved `<mask>`.** Build it from a [`mode="mlm"` corpus](../data/lm-corpora.md#mlm-mode)
  via `MLMTransformer.from_corpus(corpus, ...)`; the corpus reserves the `<mask>` piece and the
  [`MLMDataset`](../../reference/core.md) applies the dynamic 80/10/10 masking.

See [Pretrain a masked LM](../../how-to/pretrain-masked-lm.md).

---

Building a variant of one of these, or something new? See the reusable
[Building blocks](building-blocks.md) and [Writing your own model](custom-models.md). Full
signatures live in the [API reference](../../reference/core.md).
