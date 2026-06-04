# Train a language model

Train a decoder-only LM end to end — build a corpus, pack it, fit a `GPT`, measure
perplexity, and sample. This is the recipe; for the narrated walkthrough see
[Quickstart: train a GPT](../get-started/quickstart-gpt.md).

```python
from autonmt.datasets.lm_corpus import LMCorpusBuilder
from autonmt.backends import LMTrainer
from autonmt.core.nn.models import GPT
from autonmt.core.decoding import TopPSampling

# 1. Build + pack the corpus (one document per line).
builder = LMCorpusBuilder(
    base_path="data",
    corpus=[{
        "name": "wikitext", "mode": "text", "sizes": [("original", None)],
        "text": open("corpus.txt").read().splitlines(),
    }],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [16000]}],
    val_size=0.05,
).build()
corpus = builder.get_train_ds()[0]

# 2. Size a GPT to the corpus tokenizer.
model = GPT.from_corpus(corpus, embed_dim=512, num_layers=8, num_heads=8, max_seq_len=512)

# 3. Train, evaluate, generate.
trainer = LMTrainer.from_corpus(corpus, run_prefix="lm", model=model)
trainer.fit(corpus, block_size=512, max_epochs=10, batch_size=32,
            learning_rate=3e-4, scheduler="noam", warmup_steps=2000, seed=42)

print(trainer.evaluate(corpus)["ppl"])
print(trainer.generate("Once upon a time", sampler=TopPSampling(top_p=0.9), max_new_tokens=64))
```

## Notes that matter

- **`block_size` ≤ `max_seq_len`.** The packing window can't exceed the model's context bound.
  Bigger `block_size` = more context per step and more memory.
- **Scale by editing `GPT(...)`.** `embed_dim`, `num_layers`, `num_heads` are the size dials;
  the [catalog](../guide/models/catalog.md#gpt) lists them. Defaults build a small model.
- **Sweep like any grid.** Add `vocab_sizes`/`subword_models` to `encoding`, or declare several
  corpora, and loop over `builder.get_train_ds()` — one `LMTrainer` per cell — exactly as in
  [Compare multiple models](compare-models.md).
- **Resuming is automatic.** `fit` skips a run whose checkpoints already exist unless
  `force_overwrite=True`; `evaluate` / `generate` can `load_checkpoint("best")` in a fresh
  process.
- **Real data on disk.** Instead of `text=[...]`, drop a `data.txt` under the corpus's
  `data/0_raw/` and omit the inline field — see [LM corpora](../guide/data/lm-corpora.md#bring-your-own-data).
- **Pretrained instead of from scratch?** To fine-tune a Hub checkpoint (GPT-2, Llama…)
  rather than train `GPT` from zero, swap `LMTrainer` for
  [`HuggingFaceCausalLM`](../guide/backends/huggingface.md#language-models) — same
  `fit` / `evaluate` / `generate`.

---

To train on `(prompt, completion)` pairs instead, see
[Instruction-tune a model](instruction-tune.md).
