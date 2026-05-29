# Philosophy

AutoNMT exists because most of the effort in a translation experiment has nothing to do
with the idea you're testing. You spend it cleaning corpora, training tokenizers, wiring
DataLoaders, writing checkpoint and logging boilerplate, decoding test sets, running
metrics, and assembling tables. AutoNMT's job is to make that machinery disappear so the
interesting variable is the only thing you touch.

A few principles shape every design decision in the framework.

## 1. Declare the experiment, don't script it

You don't write a `for` loop over vocab sizes and copy-paste training code. You **declare
the axes** — datasets, language pairs, training sizes, subword models, vocabulary sizes —
and AutoNMT unrolls the cross-product into concrete dataset cells. Each cell is a fully
materialized variant you can train and compare.

The payoff is that the result is *one comparable table*, not a directory of ad-hoc runs
you have to reconcile by hand. See [The grid](grid.md).

## 2. Everything is path-driven and persisted

Every stage of the pipeline writes its output to a **numbered folder on disk** before the
next stage reads it. Splits, normalized text, the SentencePiece model, encoded data,
checkpoints, decoded hypotheses, and scores all live somewhere you can open and inspect.

Two consequences fall out of this:

- **Resumability.** Each stage checks `force_overwrite` and skips work that already
  exists. Re-running an experiment after a crash continues where it stopped. When you need
  to redo a stage, you delete *that stage's folder* — not the whole tree.
- **Inspectability.** When a result looks wrong, you can read the actual intermediate
  files. There is no hidden in-memory state that only existed during the run.

See [On-disk layout](on-disk-layout.md).

## 3. One surface, swappable backends

Training an AutoNMT Lightning model, fine-tuning a HuggingFace checkpoint, and shelling
out to Fairseq are very different operations. AutoNMT hides them behind a single
`fit()` / `predict()` contract so the surrounding experiment code is identical. You change
*one class* to switch engines. See [Backends](../backends/index.md).

## 4. Minimal core, extend at the edges

The built-ins stay small and predictable. Rather than growing a hundred configuration
flags, AutoNMT gives you **extension points**: callable hooks (`preprocess_raw_fn`,
`preprocess_splits_fn`, `preprocess_fn`) for data, and subclassing (`LitSeq2Seq`) for
models. If a behavior is specific to your project, you supply it as a function or a
subclass instead of waiting for a flag.

This keeps the framework legible: you can read the core, and the parts that are *yours*
stay in *your* code.

## 5. Reproducible by construction

Because every artifact is persisted and every run dumps its full effective config, an
experiment is auditable after the fact. A single `manual_seed` seeds Python, NumPy, Torch,
and Lightning together, and AutoNMT builds on widely-used reference tools (SentencePiece,
sacreBLEU, Moses, COMET, BERTScore) so numbers are comparable across papers. See
[Reproducibility](reproducibility.md).

---

Next: see how these principles assemble into [the three-layer pipeline](pipeline.md).
