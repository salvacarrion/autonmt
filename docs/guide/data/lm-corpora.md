# LM corpora: single-stream & instruct

Translation needs a _parallel_ corpus — source and target, line-aligned. A **language
model** needs only one stream of text. `LMCorpusBuilder` is the single-stream sibling of the
[`ParallelCorpusBuilder`](datasets.md): it owns the input side of the **LM** pipeline — train a
tokenizer, then pack the text into fixed-length token blocks ready for a
[`GPT`](../models/catalog.md#gpt). It does **not** replace `ParallelCorpusBuilder`; the parallel
path is untouched.

```python
from autonmt.datasets import LMCorpusBuilder

builder = LMCorpusBuilder(
    base_path="data",
    corpus=[{
        "name": "wikitext",
        "mode": "text",                       # single-stream language modelling
        "sizes": [("original", None)],
        "text": open("corpus.txt").read().splitlines(),   # one document per line
    }],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [16000]}],
    val_size=0.1,
).build()

corpus = builder.get_train_ds()[0]
```

The `encoding` axis is declared exactly as for parallel datasets — same
[subword tokenization](tokenization.md) machinery, same cross-product unrolling — so a
sweep over `vocab_sizes` or `subword_models` works the same way it does for translation.

## Two modes

A corpus declares one of three modes:

| Mode         | Source                                       | What the model learns                                      |
| ------------ | -------------------------------------------- | ---------------------------------------------------------- |
| `"text"`     | one document per line (`text=[...]`)         | predict every next token (decoder-only LM / pretraining)   |
| `"instruct"` | `(prompt, completion)` pairs (`pairs=[...]`) | predict only the completion (instruction tuning)           |
| `"mlm"`      | one document per line (`text=[...]`)         | predict randomly masked tokens (encoder-only / BERT-style) |

### Text mode

Each line is a document. The builder wraps it as `<s> … </s>`, concatenates everything into
one stream, and packs it. Every position contributes to the loss.

### Instruct mode

```python
corpus=[{
    "name": "qa", "mode": "instruct", "sizes": [("original", None)],
    "pairs": [("Translate to French: hello", "bonjour"), ...],
}]
```

The tokenizer is trained on **both** sides (so the prompt vocabulary is covered), and each
example is packed as `<s> prompt completion </s>` — but alongside the token stream the
builder writes a parallel **supervise mask** that is `0` over the prompt and `1` over the
completion. At training time the prompt positions are excluded from the loss, so the model
learns to _respond_, not to reproduce the instruction.

!!! info "Why mask the prompt?"
If the loss counted the prompt tokens, the model would spend capacity learning to predict
the _instruction_ — which is given at inference, not generated. Masking it (the standard
supervised-fine-tuning objective, [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155))
focuses every gradient on the part you actually want the model to produce. Pure-LM and
instruct share one training code path — the only difference is whether that mask is all
ones or zero over the prompt.

### MLM mode

```python
corpus=[{"name": "wikitext", "mode": "mlm", "sizes": [("original", None)],
         "text": open("corpus.txt").read().splitlines()}]
```

Identical to text mode on disk (single stream, packed), with one addition: the tokenizer
reserves a dedicated **`<mask>`** piece in the vocabulary. This is what an encoder-only
[`MLMTransformer`](../models/catalog.md#mlmtransformer) trains on — a _bidirectional_ model
that predicts randomly masked tokens rather than the next one. Unlike instruct, there is **no**
supervise file on disk: masking is **dynamic**, re-sampled every epoch by
[`MLMDataset`](../../reference/core.md) (the BERT 80/10/10 scheme), so the same packed corpus
yields a fresh corruption each pass. Train it with an
[`AutonmtMaskedLM`](../../reference/backends.md) and see
[Pretrain a masked LM](../../how-to/pretrain-masked-lm.md).

!!! info "Decoder-only vs encoder-only"
A **decoder-only** LM (`mode="text"`/`"instruct"`) is causal — each position sees only the
past — and _generates_ left to right. An **encoder-only** masked LM (`mode="mlm"`) is
_bidirectional_ — each position sees the whole sequence — and instead of generating, it
_fills in_ masked positions. Together with the encoder–decoder translation models, that's
the three Transformer families, all on the same pipeline
([Devlin et al., 2019](https://arxiv.org/abs/1810.04805)).

## Packing

Both modes produce a **packed** token stream rather than padded sentences.

!!! info "What's packing? (and why no padding)"
Sentences vary in length, so batching padded sentences wastes compute on `<pad>`.
**Packing** concatenates the whole tokenized corpus into one long stream and slices it
into contiguous, fixed-length windows of `block_size` tokens — no padding, every position
trains the model ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)). The window
size is a **training-time** choice (`block_size` on
[`AutonmtCausalLM.fit`](../training/training.md)), not a property of the corpus, so one packed
corpus can be trained at different context lengths.

`block_size` must be ≤ the model's `max_seq_len`. The torch dataset that does the slicing is
[`LMDataset`](../../reference/core.md); it hands the trainer `(x, y, supervise)` where `y` is
`x` shifted by one.

## Bring your own data

`text=[...]` / `pairs=[...]` are inline conveniences (handy for examples and tests). For real
corpora, either pass the file's lines as shown above, or place the raw files on disk under the
corpus's `data/0_raw/` folder and omit the inline field — the builder picks them up. Text mode
expects `data.txt`; instruct mode expects aligned `data.prompt` and `data.completion` files.

## What lands on disk

The layout mirrors the parallel pipeline's numbered stages, keyed by subword model and vocab
size:

```
data/<name>/<size>/
  data/0_raw/                       data.txt   (or data.prompt + data.completion)
  data/1_splits/                    {train,val}.txt   (split off the last val_size lines)
  data/4_encoded/<sw>/<vs>/         {train,val}.tokens.npy   (+ .sup.npy for instruct)
  vocabs/<sw>/<vs>/                 spm.model + spm.vocab
  models/<engine>/runs/<run>/       checkpoints, logs   (engine = backend's ENGINE: autonmt | huggingface)
```

As with the parallel builder, each stage checks `force_overwrite` before rewriting, so
re-running skips completed work. The corpus object exposes everything the model and trainer
need — `corpus.model_vocab_size`, `corpus.encode(...)` / `corpus.decode(...)`, and the special
token ids (`sos_id` / `eos_id` / `pad_id`).

!!! note "SentencePiece caps the vocab size"
SentencePiece can only build as many pieces as the corpus supports; on a small corpus it
will refuse a large `vocab_size` and tell you the maximum. Toy corpora therefore need small
vocabularies — see the [GPT quickstart](../../get-started/quickstart-gpt.md). On real data
this rarely bites.

---

Next: size a model to the corpus in the [Model catalog](../models/catalog.md#gpt), then
[train it](../training/training.md) and [generate](../translation/text-generation.md). Full
signatures are in the [API reference](../../reference/datasets.md#lmcorpusbuilder).
