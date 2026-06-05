# Instruction-tune a model

Train a decoder-only model on `(prompt, completion)` pairs so it learns to **respond**. The
only difference from [training a plain LM](train-language-model.md) is the corpus: switch
`mode` to `"instruct"` and provide `pairs`. The builder masks the prompt out of the loss; the
model, trainer, and generation are identical.

```python
from autonmt.datasets.lm_corpus import LMCorpusBuilder
from autonmt.backends import AutonmtCausalLM
from autonmt.core.nn.models import GPT
from autonmt.core.decoding import GreedySearch

pairs = [
    ("Translate to French: hello", "bonjour"),
    ("Translate to French: thank you", "merci"),
    # ... thousands more ...
]

# 1. Instruct corpus → packs (tokens, supervise-mask); prompt span gets mask=0.
builder = LMCorpusBuilder(
    base_path="data",
    corpus=[{"name": "ft", "mode": "instruct", "sizes": [("original", None)], "pairs": pairs}],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [8000]}],
    val_size=0.05,
).build()
corpus = builder.get_train_ds()[0]

# 2-3. Same GPT / AutonmtCausalLM as pure LM.
model = GPT.from_corpus(corpus, embed_dim=256, num_layers=6, num_heads=8, max_seq_len=256)
trainer = AutonmtCausalLM.from_corpus(corpus, run_prefix="ft", model=model)
trainer.fit(corpus, block_size=256, max_epochs=10, batch_size=32, learning_rate=3e-4, seed=42)

# 4. Generate with the SAME prompt shape used in training.
print(trainer.generate("Translate to French: good morning", strategy=GreedySearch()))
```

## What's different from pure LM

- **The loss is completion-only.** The builder writes a `supervise` mask that is `0` over the
  prompt and `1` over the completion, so gradients focus on the response
  ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). This is handled for you — see
  [LM corpora → instruct mode](../guide/data/lm-corpora.md#instruct-mode).
- **Prompt formatting is part of the contract.** The model learns to continue a *specific*
  prompt shape. Generate with the same template (and the same leading `<s>`, which `generate`
  adds by default) you trained on, or quality drops.
- **Perplexity reflects responses.** [`evaluate`](../guide/evaluation/perplexity.md) averages
  loss over the completion tokens only, so it measures how well the model predicts answers.

## Starting from a pretrained checkpoint

To instruction-tune a model you already pretrained (rather than from scratch), load its weights
before `fit`:

```python
model = GPT.from_corpus(corpus, embed_dim=256, num_layers=6, num_heads=8, max_seq_len=256)
trainer = AutonmtCausalLM.from_corpus(corpus, run_prefix="ft", model=model)
trainer.load_checkpoint("/path/to/pretrained/best.pt")   # warm start
trainer.fit(corpus, block_size=256, max_epochs=3, learning_rate=1e-4)
```

Use a smaller learning rate and fewer epochs than pretraining. The pretrained model must share
this corpus's tokenizer/vocabulary.

---

A runnable version is in
[`examples/03_llm/02_instruct.py`](https://github.com/salvacarrion/autonmt/blob/main/examples/03_llm/02_instruct.py).
