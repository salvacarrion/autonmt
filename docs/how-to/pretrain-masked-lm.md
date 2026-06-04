# Pretrain a masked LM

Train an encoder-only, BERT-style masked language model: build a `mode="mlm"` corpus, fit an
`MLMTransformer`, measure masked-token accuracy, and fill in blanks. The only differences from
[training a decoder-only LM](train-language-model.md) are the corpus mode, the model class, and
the trainer — the data pipeline and PyTorch-Lightning scaffolding are shared.

```python
from autonmt.datasets.lm_corpus import LMCorpusBuilder
from autonmt.backends import MLMTrainer
from autonmt.core.nn.models import MLMTransformer

# 1. MLM corpus: single stream like "text", but it reserves a <mask> token.
builder = LMCorpusBuilder(
    base_path="data",
    corpus=[{
        "name": "wikitext", "mode": "mlm", "sizes": [("original", None)],
        "text": open("corpus.txt").read().splitlines(),
    }],
    encoding=[{"subword_models": ["bpe"], "vocab_sizes": [16000]}],
    val_size=0.05,
).build()
corpus = builder.get_train_ds()[0]

# 2. A bidirectional encoder, sized to the corpus tokenizer.
model = MLMTransformer.from_corpus(corpus, embed_dim=512, num_layers=8, num_heads=8, max_seq_len=512)

# 3. Train (masks ~15% of tokens per batch, dynamically), evaluate, fill masks.
trainer = MLMTrainer.from_corpus(corpus, run_prefix="mlm", model=model, mlm_prob=0.15)
trainer.fit(corpus, block_size=512, max_epochs=10, batch_size=32, learning_rate=1e-4, seed=42)

print(trainer.evaluate(corpus)["masked_acc"])
print(trainer.fill_mask("the <mask> sat on the mat", top_k=5))
```

## Notes that matter

- **`mode="mlm"` reserves `<mask>`.** It packs the stream exactly like `"text"`, but trains the
  tokenizer with a dedicated `<mask>` piece. There is no supervise file on disk — masking is
  dynamic (re-sampled each epoch) and lives in [`MLMDataset`](../guide/data/lm-corpora.md#mlm-mode).
- **`mlm_prob`** (default `0.15`) is the fraction of tokens corrupted per block, following the
  BERT 80/10/10 scheme (80% `<mask>`, 10% random, 10% unchanged).
- **No generation.** An MLM is bidirectional and doesn't decode left to right; use
  `fill_mask("… <mask> …")` instead of `generate`. Put the literal `<mask>` where you want a
  prediction.
- **Evaluation is masked-token accuracy**, not perplexity over the whole stream — see
  [Perplexity & LM evaluation](../guide/evaluation/perplexity.md#masked-language-models).
- **Scale by editing `MLMTransformer(...)`** and sweep like any other grid by adding axes to
  `encoding` or declaring multiple corpora.

A runnable, offline version is in
[`examples/03_llm/03_masked_lm.py`](https://github.com/salvacarrion/autonmt/blob/main/examples/03_llm/03_masked_lm.py).
