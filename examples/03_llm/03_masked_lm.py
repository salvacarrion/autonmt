"""
============================================================================
 Tutorial 03.03 — Masked language model (BERT-style, encoder-only)
============================================================================

What you'll learn
-----------------
The third model family, completing the trio: encoder–decoder (translation),
decoder-only (`GPT`), and now **encoder-only** masked LM. Same pipeline, a
different objective:

    1. An `LMCorpusBuilder` with `mode="mlm"` — single-stream text, but it
       reserves a `<mask>` token in the vocabulary.
    2. An `MLMTransformer` (bidirectional encoder) trained with an `AutonmtMaskedLM`.
    3. `evaluate()` (masked-token accuracy) + `fill_mask()` (predict the token
       at a `<mask>`).

What's different vs the GPT tutorials
-------------------------------------
- The model is **bidirectional** (every position sees the whole sequence) and
  there's no left-to-right generation. Instead of `generate`, you `fill_mask`.
- Training corrupts ~15% of tokens and predicts the originals (the BERT 80/10/10
  scheme), re-sampled every epoch. Masking is handled for you by `MLMDataset`.
- Evaluation is masked-token accuracy, not perplexity over the whole stream.

This uses a tiny synthetic corpus with simple `DET ADJ NOUN VERB DET ADJ NOUN`
structure and **word-level** tokenization, so after a few epochs `fill_mask`
on `"the <mask> fox ..."` should predict an adjective. Runs offline in seconds.

Run
---
    pip install -e .
    python examples/03_llm/03_masked_lm.py
"""
import random

from autonmt.backends import AutonmtMaskedLM
from autonmt.core.nn.models import MLMTransformer
from autonmt.datasets.lm_corpus import LMCorpusBuilder

BASE_PATH = "datasets/03_masked_lm"
CORPUS = "toy_lang"

DET = ["the", "a"]
ADJ = ["quick", "lazy", "bright", "green", "small"]
NOUN = ["fox", "dog", "moon", "river", "stone"]
VERB = ["jumps", "runs", "sees", "follows", "hides"]


def make_sentence(rng):
    return " ".join([
        rng.choice(DET), rng.choice(ADJ), rng.choice(NOUN), rng.choice(VERB),
        rng.choice(DET), rng.choice(ADJ), rng.choice(NOUN),
    ]) + " ."


def main():
    rng = random.Random(42)
    text_lines = [make_sentence(rng) for _ in range(3000)]

    # 1. Build an MLM corpus: word-level tokenizer (+ reserved <mask>), packed.
    builder = LMCorpusBuilder(
        base_path=BASE_PATH,
        corpus=[{
            "name": CORPUS,
            "mode": "mlm",                  # encoder-only masked LM
            "sizes": [("original", None)],
            "text": text_lines,
        }],
        # Word-level so each token is a whole word → fill_mask predicts words.
        # With a word model the vocab equals the number of distinct words, so size
        # it to the toy's vocabulary (18 words + unk/sos/eos/pad/<mask> = 23).
        encoding=[{"subword_models": ["word"], "vocab_sizes": [23]}],
        val_size=0.1,
    ).build(force_overwrite=False)

    corpus = builder.get_train_ds()[0]

    # 2. A small bidirectional encoder, sized to the corpus tokenizer.
    model = MLMTransformer.from_corpus(
        corpus, embed_dim=128, num_layers=4, num_heads=4, max_seq_len=64, dropout=0.1,
    )

    # 3. Train (masks ~15% of tokens per batch, dynamically).
    trainer = AutonmtMaskedLM.from_corpus(corpus, run_prefix="mlm", model=model, mlm_prob=0.15)
    trainer.fit(corpus, block_size=32, max_epochs=25, batch_size=64,
                learning_rate=1e-3, seed=42, accelerator="auto")

    # 4. Masked-token accuracy on the held-out split.
    metrics = trainer.evaluate(corpus, block_size=32)
    print(f"\nMasked-token accuracy: {metrics['masked_acc']:.3f}\n")

    # 5. Fill in the blanks. Put a literal <mask> where you want a prediction.
    #    (Use only words the toy knows; sentences follow DET ADJ NOUN VERB DET ADJ NOUN.)
    for sentence in ["the <mask> fox jumps a lazy dog .",
                     "a quick <mask> sees the small river ."]:
        preds = trainer.fill_mask(sentence, top_k=3)
        print(f"  {sentence}")
        print(f"    -> top-3 at <mask>: {preds[0] if preds else '(no mask found)'}")


if __name__ == "__main__":
    main()
