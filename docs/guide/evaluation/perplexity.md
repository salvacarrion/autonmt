# Perplexity & LM evaluation

A translation model is scored against a reference with [BLEU/chrF/COMET](metrics.md). A
language model has no reference — so its intrinsic metric is **perplexity** on a held-out
split. [`AutonmtCausalLM.evaluate`](../../reference/backends.md) computes it:

```python
metrics = trainer.evaluate(corpus, split="val", block_size=64)
# {"loss": 3.09, "ppl": 21.88, "tokens": 269}
```

It returns the mean per-token cross-entropy (`loss`), its exponential (`ppl`), and the number
of supervised tokens it was averaged over.

## What perplexity is

!!! info "Perplexity in one paragraph"
A language model assigns a probability to each next token. Its quality on a held-out
sequence is the average **cross-entropy** (negative log-likelihood) it pays per token;
**perplexity** is the exponential of that:

    $$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(x_i \mid x_{<i})\right)$$

    Intuitively it's *how many equally-likely choices the model feels it faces at each step* —
    a model certain of the next token has low perplexity; a confused one, high. **Lower is
    better.** A uniform guess over a vocabulary of size $V$ gives $\text{PPL} = V$; any useful
    model scores far below that.

## How AutoNMT computes it

`evaluate` packs the chosen split into the same fixed-length blocks used for training, runs the
model, and averages cross-entropy over the **supervised** positions:

- For a **text** corpus, every position is supervised — standard LM perplexity.
- For an **instruct** corpus, only the **completion** positions count (the prompt is masked,
  exactly as in [training](../data/lm-corpora.md#instruct-mode)), so the number measures how
  well the model predicts _responses_, not instructions.

That's why the result reports `tokens`: the denominator $N$ is the count of supervised tokens,
not raw positions.

!!! warning "Perplexity only compares like with like"
Perplexity depends on the **tokenizer and vocabulary**: a model over 4 000 BPE pieces and
one over 32 000 are not directly comparable, because they're predicting over different
units. Compare perplexities only across models that share the _same_ tokenizer and the
_same_ evaluation split. To compare across tokenizers, fall back on a downstream task or a
bits-per-character normalization.

## Perplexity isn't everything

Perplexity measures next-token likelihood, not whether generations are _useful_. Two
complements:

- **Eyeball samples.** [Generate](../translation/text-generation.md) from a few held-out
  prompts and read the output — especially for instruct models, where the goal is a correct
  response, not low loss.
- **Score a downstream task.** If your LM produces something checkable (translations, answers,
  reversed sequences…), score it with the regular [metric backends](metrics.md) and roll it
  into a [report](reports.md), exactly as for translation.

## Masked language models

An encoder-only [masked LM](../data/lm-corpora.md#mlm-mode) is evaluated differently:
[`AutonmtMaskedLM.evaluate`](../../reference/backends.md) reports **masked-token accuracy** (the
fraction of masked positions the model recovers) alongside a pseudo-perplexity over those
positions:

```python
metrics = mlm_trainer.evaluate(corpus)        # {"loss", "ppl", "masked_acc", "tokens"}
```

Because masking is dynamic (re-sampled each call), the number wobbles slightly run-to-run. For
a qualitative check, [`fill_mask`](../translation/text-generation.md#masked-language-models)
predicts the tokens at `<mask>` positions directly.

---

For the generation side, see [Text generation & sampling](../translation/text-generation.md);
for reference-based metrics, [Metrics](metrics.md).
