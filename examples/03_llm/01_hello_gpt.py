"""
============================================================================
 Tutorial 03.01 — Hello GPT (decoder-only language model)
============================================================================

What you'll learn
-----------------
The LM counterpart of `01_hello_autonmt.py`. Same three-block shape, but for a
*single-stream* language model instead of translation:

    1. An `LMCorpusBuilder` that trains a tokenizer and packs the text into
       fixed-length token blocks (nanoGPT-style).
    2. A `GPT` (decoder-only Transformer) trained with an `AutonmtCausalLM`.
    3. `evaluate()` (perplexity) + `generate()` (sample a continuation).

What's different vs the translation tutorials
---------------------------------------------
- There is no language pair and no parallel target: an LM corpus is one stream
  of text. `LMCorpus`/`LMCorpusBuilder` live alongside `ParallelCorpus` — they
  don't replace it.
- The model is a `GPT` (no encoder, no cross-attention). Training reuses the
  same PyTorch-Lightning machinery via `AutonmtCausalLM`.
- Evaluation is perplexity, not BLEU.

This script uses a tiny *synthetic* corpus with simple sentence structure so it
runs offline in seconds. To use real data, replace `text=[...]` with the lines
of your own corpus (one document per line) or point the builder at raw files.

Run
---
    pip install -e .
    python examples/03_llm/01_hello_gpt.py

Expected output
---------------
A minute of training logs, a perplexity number, and a few sampled sentences.
On this toy corpus the samples should look like plausible (if nonsensical)
sentences drawn from the template — proof the pipeline learns and generates.
"""
import random

from autonmt.backends import AutonmtCausalLM
from autonmt.core.decoding import TopPSampling
from autonmt.core.nn.models import GPT
from autonmt.datasets import LMCorpusBuilder

BASE_PATH = "datasets/03_hello_gpt"
CORPUS = "toy_lang"

# A toy "language": each sentence follows  DET ADJ NOUN VERB DET ADJ NOUN .
# Enough structure that a tiny GPT learns the n-gram regularities quickly.
DET = ["the", "a"]
ADJ = ["quick", "lazy", "bright", "green", "small"]
NOUN = ["fox", "dog", "moon", "river", "stone"]
VERB = ["jumps over", "runs past", "sees", "follows", "hides from"]


def make_sentence(rng):
    return " ".join([
        rng.choice(DET), rng.choice(ADJ), rng.choice(NOUN), rng.choice(VERB),
        rng.choice(DET), rng.choice(ADJ), rng.choice(NOUN),
    ]) + " ."


def main():
    rng = random.Random(42)
    text_lines = [make_sentence(rng) for _ in range(2000)]

    # 1. Build the corpus: trains a BPE-256 tokenizer and packs the stream into
    #    blocks. Re-running skips stages already on disk.
    builder = LMCorpusBuilder(
        base_path=BASE_PATH,
        corpus=[{
            "name": CORPUS,
            "mode": "text",                 # single-stream language modelling
            "sizes": [("original", None)],
            "text": text_lines,             # inline source (written to 0_raw)
        }],
        # Small vocab: the toy template has few distinct subwords, so keep the
        # BPE size modest (SentencePiece caps it at the achievable maximum).
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [128]}],
        val_size=0.1,
    ).build(force_overwrite=False)

    corpus = builder.get_train_ds()[0]

    # 2. A small decoder-only Transformer. `from_corpus` infers vocab size /
    #    pad id from the corpus's tokenizer.
    model = GPT.from_corpus(
        corpus,
        embed_dim=128, num_layers=4, num_heads=4,
        max_seq_len=128, dropout=0.1,
    )

    trainer = AutonmtCausalLM.from_corpus(corpus, run_prefix="hello", model=model)
    trainer.fit(
        corpus,
        block_size=64,
        max_epochs=5, batch_size=64, learning_rate=3e-3, seed=42,
        accelerator="auto",
    )

    # 3. Perplexity on the held-out split, then a few sampled continuations.
    metrics = trainer.evaluate(corpus, block_size=64)
    print(f"\nValidation perplexity: {metrics['ppl']:.2f}\n")

    strategy = TopPSampling(top_p=0.9, temperature=0.8)
    for prompt in ["the quick", "a lazy dog", "the bright moon"]:
        out = trainer.generate(prompt, strategy=strategy, max_new_tokens=16)
        print(f"  {prompt!r:22} -> {out!r}")


if __name__ == "__main__":
    main()
