"""
============================================================================
 Advanced 03 — Catastrophic forgetting & weight-regularisation
============================================================================

What you'll learn
-----------------
When you fine-tune a trained model on a *new* distribution, the weights drift
away from what made the previous distribution work — and BLEU on the old data
collapses. This is **catastrophic forgetting** (McCloskey & Cohen, 1989;
Kirkpatrick et al., 2017, arXiv:1612.00796 for EWC).

This tutorial wires three weight-anchoring regularisers into AutoNMT via the
model's ``regularization_fn`` hook, and compares how well each preserves the
old task while learning the new one:

    none  →   L = L_task                                            (baseline drift)
    L1    →   L = L_task + α · Σ_i |θ_i − θ_A_i|
    L2    →   L = L_task + α · Σ_i (θ_i − θ_A_i)²
    EWC   →   L = L_task + α · Σ_i F_A_i · (θ_i − θ_A_i)²           (Fisher-weighted L2)

where θ_A are the weights at the end of Task A and F_A is the empirical
Fisher information matrix at θ_A (intuition: F_A_i is large where moving
parameter i hurt Task A's likelihood the most — so EWC penalises drift more
heavily exactly there).

Setup
-----
We split multi30k de→en into two disjoint halves by sentence length:

    Task A  =  short sentences  (src tokens <= LENGTH_THRESHOLD)
    Task B  =  long sentences   (src tokens >  LENGTH_THRESHOLD)

Both halves share the same SPM vocab — so the model architecture is identical
across tasks. We:

    1. Train on Task A only.
    2. Snapshot θ_A; compute the empirical Fisher F_A on Task A.
    3. For each reg in {none, l1, l2, ewc}, reload θ_A and fine-tune on Task B.
    4. Evaluate every fine-tuned model on (Task A test, Task B test, full test).

A successful run shows: ``none`` recovers Task B but drops on Task A;
``l2``/``ewc`` give a better Task-A/B trade-off. (The numerical gap on
multi30k with 1–2 epochs will be modest — the *mechanism* is the point.)

Where this is real CF: replace the length-based split with a real
distribution shift (different language pair, different domain, etc.) and
crank ``ALPHA`` until you see a useful Pareto front.

Run
---
    pip install -e '.[hf]'
    python examples/02_advanced/03_catastrophic_forgetting.py
"""
import datetime
import os

import torch
from torch.utils.data import DataLoader
from tokenizers.normalizers import NFKC, Strip

from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.data.translation_dataset import TranslationDataset
from autonmt.core.nn.models import Transformer
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.datasets.preprocessing import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.reporting.report import Report

BASE_PATH = "datasets/adv_03_cf"
DATASET = "multi30k"
LANG_PAIR = "de-en"

# Tokens, not characters. Multi30k descriptions average ~12 src tokens with
# BPE-4000, so 12 splits the corpus roughly in half.
LENGTH_THRESHOLD = 12

# Regularisation strength. The right scale depends on the loss magnitude and
# parameter count; 1e-3 is a reasonable starting point for L2/EWC on a tiny
# Transformer + cross-entropy ≈ O(1). For L1 it usually needs to be smaller
# (paths are linear, gradients don't decay) — see EWC paper for a discussion.
ALPHA = 1e-3


# ---------------------------------------------------------------------------
# (1) Task splits via filter_fn on the encoded data
# ---------------------------------------------------------------------------
# `filter_fn` is called by `TranslationDataset.__init__` with (src_lines,
# tgt_lines) of *encoded* text — i.e. space-separated token IDs as strings.
# We just count tokens on the source side.
def _split_by_length(src_lines, tgt_lines, keep_short: bool):
    pairs = [(s, t) for s, t in zip(src_lines, tgt_lines)
             if (len(s.split()) <= LENGTH_THRESHOLD) == keep_short]
    if not pairs:
        return [], []
    src, tgt = zip(*pairs)
    return list(src), list(tgt)


def _short(src_lines, tgt_lines, **_):
    return _split_by_length(src_lines, tgt_lines, keep_short=True)


def _long(src_lines, tgt_lines, **_):
    return _split_by_length(src_lines, tgt_lines, keep_short=False)


def _identity(src_lines, tgt_lines, **_):
    return src_lines, tgt_lines


# (name, fn) tuples — the `name` becomes the eval-subset label in the report.
TASK_A = ("taskA_short", _short)
TASK_B = ("taskB_long", _long)
ALL = ("all", _identity)


# ---------------------------------------------------------------------------
# (2) Empirical Fisher information at θ_A — used by EWC
# ---------------------------------------------------------------------------
# F_i ≈ E_{(x,y) ~ D_A} [ (∂ log p(y|x) / ∂ θ_i)^2 ]. We approximate the
# expectation by averaging squared per-batch gradients over the Task-A
# training data. This is the "empirical Fisher" — cheaper than the true
# Fisher (which samples y ~ p_θ) and standard in practical EWC.
def compute_fisher(model, dataset, batch_size=64, num_batches=None):
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=dataset.collate_fn)

    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
              if p.requires_grad}

    model.eval()  # disable dropout, but grads still flow
    seen = 0
    for i, batch in enumerate(loader):
        (x, y), _ = batch
        x, y = x.to(device), y.to(device)

        model.zero_grad()
        logits = model.forward_enc_dec(x=x, x_len=None, y=y[:, :-1], y_len=None)
        # CE matches the training loss; ignore <pad>.
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), y[:, 1:],
            ignore_index=model.padding_idx,
        )
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach().pow(2)
        seen += 1
        if num_batches is not None and seen >= num_batches:
            break

    model.zero_grad()
    return {n: f / max(seen, 1) for n, f in fisher.items()}


# ---------------------------------------------------------------------------
# (3) Regularisers — closures that get attached to ``model.regularization_fn``
# ---------------------------------------------------------------------------
# The seq2seq base calls ``regularization_fn(model, loss)`` after computing
# the task loss and BEFORE backward. We mutate ``loss`` in place (``+=`` on a
# non-leaf tensor preserves the autograd graph), so the penalty contributes
# to the gradient.
def make_l1(theta_anchor, alpha):
    def reg_fn(model, loss):
        penalty = 0.0
        for name, p in model.named_parameters():
            if name in theta_anchor:
                penalty = penalty + (p - theta_anchor[name]).abs().sum()
        loss += alpha * penalty
    return reg_fn


def make_l2(theta_anchor, alpha):
    def reg_fn(model, loss):
        penalty = 0.0
        for name, p in model.named_parameters():
            if name in theta_anchor:
                penalty = penalty + (p - theta_anchor[name]).pow(2).sum()
        loss += alpha * penalty
    return reg_fn


def make_ewc(theta_anchor, fisher, alpha):
    def reg_fn(model, loss):
        penalty = 0.0
        for name, p in model.named_parameters():
            if name in theta_anchor and name in fisher:
                penalty = penalty + (fisher[name] * (p - theta_anchor[name]).pow(2)).sum()
        loss += alpha * penalty
    return reg_fn


# ---------------------------------------------------------------------------
# (4) Standard AutoNMT preprocessing
# ---------------------------------------------------------------------------
def normalize(lines):
    return normalize_lines(lines, seq=[NFKC(), Strip()])


def preprocess_train(data, ds):
    return preprocess_pairs(data["src"]["lines"], data["tgt"]["lines"], normalize_fn=normalize)


def preprocess_predict(data, ds):
    return preprocess_lines(data["lines"], normalize_fn=normalize)


def main():
    download_hf_dataset(
        hf_id="bentrevett/multi30k", base_path=BASE_PATH,
        dataset_name=DATASET, lang_pair=LANG_PAIR,
        src_field="de", tgt_field="en",
    )

    builder = DatasetBuilder(
        base_path=BASE_PATH,
        datasets=[{
            "name": DATASET,
            "languages": [LANG_PAIR],
            "sizes": [("original", None)],
        }],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [4000]}],
        preprocess_raw_fn=preprocess_train,
        preprocess_splits_fn=preprocess_train,
        merge_vocabs=False,
    ).build(force_overwrite=False)

    train_ds = builder.get_train_ds()[0]
    test_datasets = builder.get_test_ds()
    src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)

    pred_cfg = PredictConfig(
        metrics={"bleu"}, beams=[5],
        load_checkpoint="best",
        preprocess_fn=preprocess_predict,
        eval_mode="compatible",
    )

    # -----------------------------------------------------------------------
    # (5) Phase 1 — train on Task A (short sentences) only
    # -----------------------------------------------------------------------
    print("\n=== Phase 1: train on Task A (short sentences) ===")
    model_a = Transformer.from_vocabs(src_vocab, tgt_vocab)
    trainer_a = AutonmtTranslator(
        model=model_a, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        runs_dir=train_ds.get_runs_path(toolkit="autonmt"),
        run_name=train_ds.get_run_name(run_prefix="cf_taskA"),
        train_subset=TASK_A,
        # Per-task BLEU on the test set; val stays single-dataloader so the
        # default `monitor="val_loss"` works without `/dataloader_idx_N` suffixes.
        test_subsets=[TASK_A, TASK_B, ALL],
    )
    trainer_a.fit(
        train_ds,
        config=FitConfig(max_epochs=2, batch_size=128, learning_rate=1e-3, seed=42),
    )
    scores_a_only = trainer_a.predict(test_datasets, config=pred_cfg)

    # -----------------------------------------------------------------------
    # (6) Snapshot θ_A and compute Fisher F_A
    # -----------------------------------------------------------------------
    # Important: copy/detach so the anchor doesn't track grad and doesn't get
    # mutated when we fine-tune. Use the same device as the model to avoid
    # per-step host↔device transfers inside the regulariser.
    device = next(trainer_a.model.parameters()).device
    # state_dict() includes buffers — needed for a clean `load_state_dict`.
    theta_a_full = {n: v.detach().clone().cpu()
                    for n, v in trainer_a.model.state_dict().items()}
    # named_parameters() are what the regularisers iterate — only trainables.
    theta_a = {n: p.detach().clone().to(device)
               for n, p in trainer_a.model.named_parameters()}

    # Build a Task-A train dataset just for Fisher estimation — we want the
    # same distribution the model trained on.
    train_path = train_ds.get_encoded_path(fname=train_ds.train_name)
    task_a_tds = TranslationDataset(
        file_prefix=train_path, src_lang=train_ds.src_lang,
        tgt_lang=train_ds.tgt_lang, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        filter_fn=_short,
    )
    print(f"\n=== Computing Fisher on {len(task_a_tds)} Task-A examples ===")
    fisher_a = compute_fisher(trainer_a.model, task_a_tds,
                              batch_size=64, num_batches=50)

    # -----------------------------------------------------------------------
    # (7) Phase 2 — fine-tune on Task B with each regulariser
    # -----------------------------------------------------------------------
    regimes = {
        "none": None,
        "l1":   make_l1(theta_a, ALPHA),
        "l2":   make_l2(theta_a, ALPHA),
        "ewc":  make_ewc(theta_a, fisher_a, ALPHA),
    }

    all_scores = [scores_a_only]  # include the Task-A-only baseline in the report
    for reg_name, reg_fn in regimes.items():
        print(f"\n=== Phase 2: fine-tune on Task B with reg={reg_name} ===")
        model_b = Transformer.from_vocabs(src_vocab, tgt_vocab)
        model_b.load_state_dict(theta_a_full)
        # The hook is attached to the model instance; the trainer doesn't need
        # to know it exists.
        model_b.regularization_fn = reg_fn

        trainer_b = AutonmtTranslator(
            model=model_b, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
            runs_dir=train_ds.get_runs_path(toolkit="autonmt"),
            run_name=train_ds.get_run_name(run_prefix=f"cf_taskB_{reg_name}"),
            train_subset=TASK_B,
            test_subsets=[TASK_A, TASK_B, ALL],
        )
        # Higher LR here exaggerates the drift — without reg, Task A's score
        # should drop visibly. A real CF benchmark would tune this per regime.
        trainer_b.fit(
            train_ds,
            config=FitConfig(max_epochs=2, batch_size=128, learning_rate=3e-3, seed=42),
        )
        scores = trainer_b.predict(test_datasets, config=pred_cfg)
        all_scores.append(scores)

    # -----------------------------------------------------------------------
    # (8) Report — one row per (regime × eval_ds), grouped by run_name
    # -----------------------------------------------------------------------
    out = f".outputs/adv_03_cf/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    # One row per (regime × eval_ds). Because each regime evaluates the same
    # dataset under three `test_subsets`, the per-task BLEU lands in separate
    # `translations.<task>.beam5.*` columns — visible side by side in the table.
    report = Report.from_runs(all_scores, output_path=out).save()
    print(f"\nReport saved to: {os.path.abspath(out)}\n")
    print(report)
    print(
        "\nRead the report as: for each regime, compare the BLEU on Task A\n"
        "(retention of old knowledge) vs Task B (acquisition of new knowledge).\n"
        "Stronger regularisation → better A, worse B. Tune ALPHA to find the\n"
        "knee of the trade-off."
    )


if __name__ == "__main__":
    main()
