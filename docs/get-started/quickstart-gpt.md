# Quickstart: train a GPT

This page runs a complete, minimal **language-model** experiment end to end — **build a
corpus, pack it into blocks, train a small decoder-only Transformer, measure perplexity,
and sample text** — explaining each block as we go. It's the LM counterpart of
[Quickstart: your first translation](quickstart.md), and it reuses the same training,
decoding, and reporting machinery under the hood.

No extra dependencies are needed beyond the base install:

```bash
pip install -e .
```

We'll train on a tiny **synthetic** corpus so the script runs offline in seconds. Swapping
in a real corpus is a one-line change (see [step 1](#1-build-and-pack-the-corpus)).

!!! info "What's a decoder-only language model?"
    A translation model is an **encoder–decoder**: it reads a source sentence and writes a
    target one. A **language model** (the GPT family) drops the encoder and keeps a single
    stack that predicts the *next token* given everything so far. There's no separate input
    and output — just one stream of text the model learns to continue. The same idea, with
    a masked loss, also powers instruction-tuned assistants (see the
    [end of this page](#instruction-tuning-in-one-step)).

## The whole script

Here it is end to end; the sections below unpack each part.

```python
import random

from autonmt.datasets import LMCorpusBuilder
from autonmt.backends import AutonmtCausalLM
from autonmt.core.nn.models import GPT
from autonmt.core.decoding import TopPSampling

# 0. A tiny synthetic corpus: DET ADJ NOUN VERB DET ADJ NOUN .
rng = random.Random(42)
DET, ADJ = ["the", "a"], ["quick", "lazy", "bright", "green", "small"]
NOUN, VERB = ["fox", "dog", "moon", "river", "stone"], ["jumps over", "runs past", "sees"]
def sentence():
    return " ".join([rng.choice(DET), rng.choice(ADJ), rng.choice(NOUN), rng.choice(VERB),
                     rng.choice(DET), rng.choice(ADJ), rng.choice(NOUN)]) + " ."

# 1. Build + pack the corpus (trains a tokenizer, writes fixed-length token blocks).
builder = LMCorpusBuilder(
    base_path="data",
    corpus=[{"name": "toy_lang", "mode": "text", "sizes": [("original", None)],
             "text": [sentence() for _ in range(2000)]}],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [128]}],
    val_size=0.1,
).build()
corpus = builder.get_train_ds()[0]

# 2. A small decoder-only Transformer, sized to the corpus tokenizer.
model = GPT.from_corpus(corpus, embed_dim=128, num_layers=4, num_heads=4, max_seq_len=128)

# 3. Train.
trainer = AutonmtCausalLM.from_corpus(corpus, run_prefix="hello", model=model)
trainer.fit(corpus, block_size=64, max_epochs=5, batch_size=64, learning_rate=3e-3, seed=42)

# 4. Evaluate (perplexity) and 5. generate.
print("perplexity:", trainer.evaluate(corpus, block_size=64)["ppl"])
print(trainer.generate("the quick", strategy=TopPSampling(top_p=0.9, temperature=0.8),
                        max_new_tokens=16))
```

!!! info "Numbers and samples will be modest — that's expected"
    A small model trained for a handful of epochs on a toy corpus won't write poetry. The
    point of this script is to prove the **LM pipeline runs end to end**, not to produce a
    capable model.

## Step by step

### 1 · Build and pack the corpus

```python
builder = LMCorpusBuilder(
    base_path="data",
    corpus=[{"name": "toy_lang", "mode": "text", "sizes": [("original", None)],
             "text": [sentence() for _ in range(2000)]}],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [128]}],
    val_size=0.1,
).build()
corpus = builder.get_train_ds()[0]
```

[`LMCorpusBuilder`](../reference/datasets.md) is the single-stream sibling of the parallel
[`DatasetBuilder`](../guide/data/datasets.md) — it doesn't replace it. It trains a
[SentencePiece tokenizer](../guide/data/tokenization.md) on the text and writes a flat,
**packed** token stream to disk. The `text=[...]` field is an inline convenience: drop in
the lines of your own corpus (one document per line) and everything downstream is identical.
You can also point it at raw files on disk instead of passing `text`.

!!! info "What's packing? (and why no padding)"
    Sentences vary in length, so a naive batch pads short ones with `<pad>` — wasted
    compute. **Packing** instead concatenates the whole tokenized corpus into one long
    stream and slices it into contiguous, fixed-length windows of `block_size` tokens. No
    padding, every position trains the model, and one epoch covers
    `n_tokens / block_size` blocks. It's the standard recipe for LM pretraining. Inputs are
    the block; targets are the same block shifted by one position.

### 2 · Build the model

```python
model = GPT.from_corpus(corpus, embed_dim=128, num_layers=4, num_heads=4, max_seq_len=128)
```

[`GPT`](../reference/core.md) is a pre-norm decoder-only Transformer using the modern
[building blocks](../guide/models/building-blocks.md) — rotary position embeddings (RoPE),
RMSNorm, and SwiGLU feed-forwards, with the input embedding tied to the output projection.
`from_corpus` reads the vocabulary size and pad id from the corpus's tokenizer so you don't
repeat them. `max_seq_len` is the longest context the model can attend over; it must be
≥ the `block_size` you train with.

### 3 · Train

```python
trainer = AutonmtCausalLM.from_corpus(corpus, run_prefix="hello", model=model)
trainer.fit(corpus, block_size=64, max_epochs=5, batch_size=64, learning_rate=3e-3, seed=42)
```

[`AutonmtCausalLM`](../reference/backends.md) plays the role `AutonmtTranslator` plays for
translation, but its verbs are **`fit` / `evaluate` / `generate`** (there's no parallel
test set to translate-and-score). `from_corpus` wires the run's checkpoints and logs to the
corpus's on-disk location. `fit` reuses the same PyTorch-Lightning machinery as the
translation backend — optimizer, schedulers, checkpointing (best/last), TensorBoard — so
everything you know from [Training a model](../guide/training/training.md) carries over.
`block_size` is the context length each training block spans.

### 4 · Evaluate (perplexity)

```python
metrics = trainer.evaluate(corpus, block_size=64)   # {"loss": ..., "ppl": ..., "tokens": ...}
```

There's no BLEU here — an LM has no reference to compare against. The intrinsic metric is
**perplexity** on the held-out split.

!!! info "What's perplexity?"
    Perplexity is `exp(average per-token cross-entropy)`: loosely, *how many equally-likely
    choices the model feels it has at each step*. A model that's certain of the next token
    has low perplexity; a confused one has high. Lower is better, and it's the standard way
    to compare language models on the same tokenizer.

### 5 · Generate

```python
text = trainer.generate("the quick", strategy=TopPSampling(top_p=0.9, temperature=0.8),
                        max_new_tokens=16)
```

`generate` tokenizes the prompt, runs **KV-cached** autoregressive decoding, and detokenizes
the result. The `strategy` is exactly the same family used for translation —
[greedy, top-k, top-p (nucleus), and multinomial](../guide/translation/decoding.md) — so
there's nothing new to learn. Pass `GreedySearch()` for deterministic output, or a sampling
strategy for variety.

## Instruction-tuning in one step

Switch `mode` to `"instruct"` and give `(prompt, completion)` pairs instead of `text`.
The builder packs a parallel **supervise mask** so the loss is computed **only over the
completion** — the model learns to *respond*, not to echo the instruction:

```python
builder = LMCorpusBuilder(
    base_path="data",
    corpus=[{"name": "reverse", "mode": "instruct", "sizes": [("original", None)],
             "pairs": [("reverse: a b c", "c b a"), ("reverse: d e f", "f e d")]}],  # ...
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [32]}],
).build()
```

Everything else — the `GPT` model, `AutonmtCausalLM`, `generate` — is unchanged; prompt masking is
entirely a data-side concern. A runnable version is in
[`examples/03_llm/02_instruct.py`](https://github.com/salvacarrion/autonmt/blob/main/examples/03_llm/02_instruct.py).

---

That's the whole LM loop. It mirrors the translation loop one-for-one — **describe the data,
build a model, `fit`, `evaluate`, `generate`** — because both ride the same pipeline. Next,
see [what landed on disk](understanding-the-output.md), or browse the
[runnable LM examples](https://github.com/salvacarrion/autonmt/tree/main/examples/03_llm).
```
