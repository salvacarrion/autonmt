"""
============================================================================
 Tutorial 03.02 — Instruction tuning (prompt → completion)
============================================================================

What you'll learn
-----------------
The *instruct* mode of the LM path: train a decoder-only model on
(prompt, completion) pairs where **only the completion contributes to the
loss**. The prompt span is masked out via the per-token `supervise` stream that
`LMCorpusBuilder` packs alongside the token ids, so the model learns to *respond*
rather than to also reproduce the instruction.

What's different vs 01_hello_gpt
--------------------------------
- `mode="instruct"` and `pairs=[(prompt, completion), ...]` instead of `text`.
- Everything else (the GPT model, `LMTrainer`, `generate`) is identical — the
  prompt masking is entirely a data-side concern.

The toy task is **sequence reversal**: given `reverse: a b c d` the target is
`d c b a`. It's deterministic, so a small model learns it fast and you can see
whether generation actually responds to the prompt.

Run
---
    pip install -e .
    python examples/03_llm/02_instruct.py

Expected output
---------------
Training logs, a perplexity number, and held-out prompts with the model's
completions. After a few epochs the reversals should be mostly correct.
"""
import random

from autonmt.backends import LMTrainer
from autonmt.core.decoding import GreedySearch
from autonmt.core.nn.models import GPT
from autonmt.datasets.lm_corpus import LMCorpusBuilder

BASE_PATH = "datasets/03_instruct"
CORPUS = "reverse"

SYMBOLS = list("abcdefgh")


def make_pair(rng):
    n = rng.randint(3, 6)
    seq = [rng.choice(SYMBOLS) for _ in range(n)]
    prompt = "reverse: " + " ".join(seq)
    completion = " ".join(reversed(seq))
    return prompt, completion


def main():
    rng = random.Random(42)
    pairs = [make_pair(rng) for _ in range(4000)]

    # 1. Build an instruct corpus. The builder trains the tokenizer on both
    #    sides and packs (tokens, supervise-mask) streams; the mask is 0 over
    #    each prompt and 1 over its completion.
    builder = LMCorpusBuilder(
        base_path=BASE_PATH,
        corpus=[{
            "name": CORPUS,
            "mode": "instruct",
            "sizes": [("original", None)],
            "pairs": pairs,
        }],
        # Tiny alphabet (8 symbols + "reverse:") → keep the BPE vocab small.
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [32]}],
        val_size=0.1,
    ).build(force_overwrite=False)

    corpus = builder.get_train_ds()[0]

    model = GPT.from_corpus(
        corpus,
        embed_dim=128, num_layers=4, num_heads=4,
        max_seq_len=128, dropout=0.1,
    )

    trainer = LMTrainer.from_corpus(corpus, run_prefix="instruct", model=model)
    trainer.fit(
        corpus,
        block_size=64,
        max_epochs=20, batch_size=64, learning_rate=3e-3, seed=42,
        accelerator="auto",
    )

    metrics = trainer.evaluate(corpus, block_size=64)
    print(f"\nValidation perplexity: {metrics['ppl']:.2f}\n")

    # Greedy decoding for a deterministic task. Held-out prompts:
    held_out = ["reverse: a b c", "reverse: d e f g", "reverse: h a h b c"]
    for prompt in held_out:
        out = trainer.generate(prompt, sampler=GreedySearch(), max_new_tokens=16)
        print(f"  {prompt!r:26} -> {out!r}")


if __name__ == "__main__":
    main()
