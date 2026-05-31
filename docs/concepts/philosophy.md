# Design philosophy

Four principles shape every API decision in AutoNMT. If you understand these, the rest of
the framework will feel predictable — because it was designed to.

## The problem it solves

A real NMT experiment is rarely *one* model on *one* dataset. It's a question — *does
byte-level encoding beat BPE on low-resource German→English, at which vocab size, and does it
hold across two test sets?* — answered by running the **same pipeline** many times, changing
one knob at a time: clean → split → tokenize → build vocab → **train** → decode → score →
tabulate. Every step except *train* is identical each time, yet that scaffolding is where the
time goes and where the silent bugs hide (a leaked test set, a vocab/checkpoint mismatch, a
BLEU computed with a different tokenizer than last week). **AutoNMT automates that repetitive
half** so the only thing you think about is the model.

## 1. Grid-first { #grid-first }

Most training code is written as **loops**: `for dataset in ...: for vocab_size in ...:
train()`. That works until the nesting grows, the bookkeeping (where do results go? which
config produced this?) leaks into the loop body, and adding one more axis means touching
code in three places.

AutoNMT inverts that. You **declare the axes** of your experiment as data, and the
framework owns the cross-product:

```python
DatasetBuilder(
    base_path="data",
    datasets=[
        {"name": "multi30k", "languages": ["de-en", "fr-en"], "sizes": [("original", None), ("50k", 50000)]},
    ],
    encoding=[
        {"subword_models": ["bpe", "unigram"], "vocab_sizes": [4000, 8000]},
    ],
)
```

That single declaration describes **2 language pairs × 2 sizes × 2 subword models × 2
vocab sizes = 16** dataset variants. AutoNMT unrolls them into 16 `Dataset` objects, each
knowing exactly where its files live on disk. Your experiment loop becomes a flat
iteration — `for ds in builder.get_train_ds(): ...` — with no nesting and no path
arithmetic.

The payoff is **comparability**: because every cell flows through the *same* code, the
numbers at the end sit in one table where the only thing that differs is the axis you
swept. That's the difference between "I ran some experiments" and "I ran a controlled
comparison."

→ See it in depth: [The mental model](mental-model.md) and [Datasets & the dataset
builder](../guide/data/datasets.md).

## 2. A minimal, extensible core { #extensible }

It is tempting, when a framework meets a new use case, to add a flag. Do that enough times
and you get a 40-argument function nobody understands and a core that's afraid to change.

AutoNMT takes the opposite bet: **keep the core small, and make extension the default
path.** Every layer is designed to be *subclassed, replaced, or hooked* rather than
configured into oblivion:

- Need a new architecture? Subclass the seq2seq base and implement
  [three methods](../guide/models/custom-models.md).
- Need a different decoding rule? Subclass [`BaseSearch`](../guide/translation/decoding.md)
  (or the one-method `BaseStepSearch`).
- Need a custom metric? Register a [`MetricBackend`](../guide/evaluation/metrics.md).
- Need bespoke data cleaning? Pass a **callable hook** (`preprocess_raw_fn`,
  `preprocess_splits_fn`, `preprocess_fn`) — you write a function, AutoNMT calls it at the
  right stage.

The rule of thumb the project follows: *if a feature can live as a subclass or a hook,
it does not become a built-in flag.* This keeps the surface you have to learn small, and
the surface you can extend large.

!!! quote "A note on `assert`"
    You'll see `assert` statements throughout the source. They're intentional — they catch
    shape and identity bugs early during research iteration (a vocab/checkpoint mismatch, a
    misaligned batch). They are part of the contract, not leftover debugging.

## 3. The toolkit abstraction — "Keras for NMT backends" { #keras }

Keras let you write one model and run it on TensorFlow, Theano, or CNTK. AutoNMT applies
the same idea one level up: **write one experiment and run it on any NMT toolkit.**

Underneath `fit()` / `predict()` is a single contract,
[`BaseTranslator`](../guide/backends/choosing.md). Three backends implement it:

| You want to…                                  | Backend                  |
| --------------------------------------------- | ------------------------ |
| Train a custom architecture from scratch      | `AutonmtTranslator`      |
| Fine-tune / evaluate a pretrained checkpoint  | `HuggingFaceTranslator`  |
| Reproduce a Fairseq baseline                  | `FairseqTranslator` *(deprecated)* |

Swapping is a one-line change — the dataset prep, the scoring, and the report are
identical across all three:

```python
# trainer = AutonmtTranslator.from_dataset(train_ds, model=..., src_vocab=..., tgt_vocab=..., run_prefix="x")
trainer = HuggingFaceTranslator.from_dataset(train_ds, model_id="Helsinki-NLP/opus-mt-de-en", run_prefix="x")
```

This is why the docs split the way they do: **data goes in and reports come out the same
way no matter the backend; only the middle swaps.**

→ How the contract works: [Choosing a backend](../guide/backends/choosing.md).

## 4. Reproducibility by construction { #reproducible }

Reproducibility in AutoNMT isn't a checklist you remember to follow — it's a side effect
of how the pipeline is built:

- **Everything is path-driven.** A `Dataset` *computes* where each stage lives. Stages are
  written to numbered folders (`0_raw`, `1_splits`, `2_preprocessed`, `4_encoded/…`), and
  each stage checks before rewriting — so a re-run **skips** completed work and a half-built
  experiment resumes instead of starting over.
- **Every run records itself.** `fit()` and `predict()` dump their full effective config to
  `logs/config_train.json` / `config_predict.json`, alongside an environment snapshot
  (Python, package versions, and the git SHA + dirty flag at run time).
- **One seed seeds everything.** `manual_seed` ties Python, NumPy, PyTorch, and Lightning
  to a single value you pass to `fit()`.

The result: months later you can look at a results folder and know exactly what produced
it — and re-running the script reconstructs only what's missing.

→ The full picture: [On-disk layout](on-disk-layout.md) and [Reproducibility
model](reproducibility.md).

---

Next: the **[mental model](mental-model.md)** — these four principles assembled into one
picture of how an experiment flows.
