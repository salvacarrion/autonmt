# Why AutoNMT

## The problem it solves

If you do research in neural machine translation, you know the shape of the work. A real
experiment is rarely *one* model on *one* dataset. It's a question:

> *Does byte-level encoding beat BPE on low-resource German→English? At which vocabulary
> size? And does the answer hold across two test sets?*

Answering that means running the **same pipeline** many times over, changing one knob at
a time:

1. Take a raw parallel corpus and clean it (normalize, filter, dedupe).
2. Split it into train / validation / test.
3. Learn a subword tokenizer and encode every split.
4. Build the vocabularies.
5. Train a model — with its own optimizer, scheduler, callbacks, logging.
6. Decode the test set with beam search.
7. Score the output with BLEU, chrF, COMET…
8. Collect all of that into a table you can actually read.

Steps 1–4 and 6–8 are **identical every time**. Only step 5 — the model — is the part
your paper is about. Yet in practice that repetitive scaffolding is where most of the
time goes, and where most of the silent bugs hide: a test set that leaked into training,
a vocabulary that didn't match the checkpoint, a BLEU score computed with a different
tokenizer than last week's, a results folder you can no longer map back to the config
that produced it.

**AutoNMT automates that repetitive half.** You describe *what* you want to compare; it
runs the cross-product, keeps every intermediate artifact in a predictable place, and
gives you back results that are comparable by construction — so the only thing left for
you to think about is the model.

## What it is (and isn't)

AutoNMT is a **research orchestration framework**, not a model zoo and not a new deep
learning library.

- It **does** own the pipeline: dataset preparation, tokenization, the training/decoding
  loop, scoring, and reporting — wired together so a whole grid of experiments is one
  script.
- It **does** ship a small, readable neural engine (PyTorch Lightning Transformers, RNNs,
  ConvS2S) and a set of decoding strategies, so you can run end-to-end out of the box.
- It **doesn't** lock you to that engine. The same script can fine-tune a HuggingFace
  checkpoint or call the Fairseq CLI — the toolkit is an implementation detail behind a
  common interface ([the "Keras for NMT toolkits" idea](philosophy.md#keras)).
- It **doesn't** try to be exhaustive. The core stays deliberately small; anything
  specialized is a [subclass or a callable hook](philosophy.md#extensible), not a new
  built-in flag.

## Who it's for

The typical user is a **researcher or graduate student** running MT experiments: someone
who needs to compare architectures, tokenizations, or data conditions rigorously and
reproducibly, and who would rather not re-implement the boilerplate (or re-discover its
bugs) for every project.

You should be comfortable with Python and the basics of training a sequence model. You do
**not** need to be an expert in every corner of NMT — these docs explain the
machine-translation concepts as they come up:

!!! info "NMT primers, inline"
    Every time a translation-specific idea appears for the first time — *subword
    tokenization*, *beam search*, *length penalty*, *teacher forcing*, *samplers* — there's
    a short callout like this one that explains the intuition (and, where it helps, the
    math) before we use it. If you already know the concept, skip the box; if you don't,
    you won't have to leave the page.

## What you get out of it

- **A grid, not a script.** Describe the axes to sweep; AutoNMT runs every combination.
- **One comparable report.** Every cell is scored the same way and collected into the
  same table and plots.
- **Reproducibility by construction.** Every stage is written to a numbered folder, every
  run dumps its full effective config plus an environment snapshot, and one seed seeds
  everything.
- **A backend you can swap.** Native Lightning models, HuggingFace fine-tuning, or Fairseq
  — same `fit` / `predict` surface.
- **A core you can extend.** New model? Subclass `LitSeq2Seq`. New decoder? Subclass
  `BaseSearch`. New metric? Register a `MetricBackend`. No forking required.

---

Next: the **[design philosophy](philosophy.md)** behind these choices, then the
**[mental model](mental-model.md)** that ties them into one picture.
