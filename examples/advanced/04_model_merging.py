"""
============================================================================
 Advanced 04 — Linear model merging (``a · θ_A + b · θ_B``)
============================================================================

What you'll learn
-----------------
The "spherical cow" of model merging: take two trained models of the same
architecture, average their weights, and see what the merged model does.

    θ_merged = α · θ_A + (1 − α) · θ_B

This works at all only because two models initialised from the same seed and
trained on related data tend to end up in the same loss basin — so a straight
line between them stays in roughly low-loss territory (Frankle et al., 2020;
Wortsman et al., 2022 "Model Soups", arXiv:2203.05482).

This tutorial:

    1. Trains two Transformers on two non-overlapping subsets of multi30k,
       seeded identically so weight averaging is meaningful.
    2. Builds a third model and overwrites its weights with the convex combo
       at α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}.
    3. Evaluates every merged model on the test set, *without retraining*.

What to expect
--------------
α=0 reproduces B's BLEU. α=1 reproduces A's. The interesting question is
what happens in between: a flat or even higher curve in the middle ⇒
both runs landed in the same basin and averaging helps (model-soup
territory); a dip ⇒ they're in different basins and a straight line cuts
through a high-loss region. With identical seed init and overlapping data
the curve here is usually flat-ish or mildly concave.

When to merge
-------------
- ``α=0.5`` of N seeds of the same recipe ⇒ a "uniform soup".
- α tuned on a held-out set ⇒ a "greedy soup" (sweep α, keep the best).
- Don't expect this to combine *capabilities* (a math model + a code model);
  for that you want task-arithmetic / TIES-merging on top of a SHARED base.

Run
---
    pip install -e '.[hf]'
    python examples/advanced/04_model_merging.py
"""
import datetime
import os

import torch
from tokenizers.normalizers import NFKC, Strip

from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.nn.models import Transformer
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.datasets.preprocessing import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.reporting.report import Report

BASE_PATH = "datasets/adv_04_merge"
DATASET = "multi30k"
LANG_PAIR = "de-en"

SEED = 42                          # same init for both runs → mergeable
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# (1) Disjoint train subsets — A keeps even-indexed lines, B keeps odd ones.
# ---------------------------------------------------------------------------
# Same distribution on each side (so the curve is meaningful), no overlap (so
# the two models have actually seen different data).
def _take(parity):
    def fn(src_lines, tgt_lines):
        pairs = [(s, t) for i, (s, t) in enumerate(zip(src_lines, tgt_lines))
                 if i % 2 == parity]
        if not pairs:
            return [], []
        src, tgt = zip(*pairs)
        return list(src), list(tgt)
    return fn


TASK_A = ("even", _take(0))
TASK_B = ("odd", _take(1))


# ---------------------------------------------------------------------------
# (2) The merge — one line of real work, the rest is plumbing
# ---------------------------------------------------------------------------
def merge_state_dicts(state_a, state_b, alpha: float):
    """Return ``alpha * state_a + (1 - alpha) * state_b`` key by key.

    Both dicts must share keys and tensor shapes. Non-floating buffers
    (e.g. integer counters) are copied from A as-is — averaging them would
    be meaningless.
    """
    assert state_a.keys() == state_b.keys(), "state_dicts must share keys"
    merged = {}
    for k, va in state_a.items():
        vb = state_b[k]
        if torch.is_floating_point(va):
            merged[k] = alpha * va + (1.0 - alpha) * vb
        else:
            merged[k] = va.clone()
    return merged


# ---------------------------------------------------------------------------
# (3) Standard AutoNMT preprocessing
# ---------------------------------------------------------------------------
def normalize(lines):
    return normalize_lines(lines, seq=[NFKC(), Strip()])


def preprocess_train(data, ds):
    return preprocess_pairs(data["src"]["lines"], data["tgt"]["lines"], normalize_fn=normalize)


def preprocess_predict(data, ds):
    return preprocess_lines(data["lines"], normalize_fn=normalize)


def train_one(train_ds, src_vocab, tgt_vocab, subset, run_prefix):
    """Train one model from a fixed seed on the given train subset."""
    model = Transformer.from_vocabs(src_vocab, tgt_vocab)
    trainer = AutonmtTranslator(
        model=model, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        runs_dir=train_ds.get_runs_path(toolkit="autonmt"),
        run_name=train_ds.get_run_name(run_prefix=run_prefix),
        train_subset=subset,
    )
    trainer.fit(
        train_ds,
        config=FitConfig(max_epochs=2, batch_size=128, learning_rate=1e-3, seed=SEED),
    )
    return trainer, model


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
    # (4) Train the two endpoints
    # -----------------------------------------------------------------------
    print("\n=== Training model A on even-indexed pairs ===")
    trainer_a, model_a = train_one(train_ds, src_vocab, tgt_vocab, TASK_A, "merge_A")

    print("\n=== Training model B on odd-indexed pairs ===")
    trainer_b, model_b = train_one(train_ds, src_vocab, tgt_vocab, TASK_B, "merge_B")

    # Pull state_dicts ONCE; merging is just arithmetic from here.
    state_a = {k: v.detach().clone().cpu()
               for k, v in model_a.state_dict().items()}
    state_b = {k: v.detach().clone().cpu()
               for k, v in model_b.state_dict().items()}

    # -----------------------------------------------------------------------
    # (5) Sweep α — evaluate the merged model without retraining
    # -----------------------------------------------------------------------
    # We re-use the same `trainer` shell each iteration; only the weights
    # change. `load_checkpoint=None` tells predict() to skip checkpoint
    # loading and trust whatever is on the in-memory model. `force_overwrite`
    # is True because each α writes a fresh hyp.txt to the same path.
    pred_cfg_merge = PredictConfig(
        metrics={"bleu"}, beams=[5],
        load_checkpoint=None,
        preprocess_fn=preprocess_predict,
        eval_mode="compatible",
        force_overwrite=True,
    )

    all_scores = []
    for alpha in ALPHAS:
        merged_state = merge_state_dicts(state_a, state_b, alpha)
        merged_model = Transformer.from_vocabs(src_vocab, tgt_vocab)
        merged_model.load_state_dict(merged_state)

        run_prefix = f"merge_a{alpha:.2f}"
        print(f"\n=== Evaluating merged model α={alpha:.2f} ===")
        trainer = AutonmtTranslator(
            model=merged_model, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
            runs_dir=train_ds.get_runs_path(toolkit="autonmt"),
            run_name=train_ds.get_run_name(run_prefix=run_prefix),
        )
        scores = trainer.predict(test_datasets, config=pred_cfg_merge)
        # Tag the row so the report makes the sweep obvious.
        for s in scores:
            s["alpha"] = alpha
        all_scores.append(scores)

    # -----------------------------------------------------------------------
    # (6) Report — BLEU vs α curve
    # -----------------------------------------------------------------------
    out = f".outputs/adv_04_merge/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    report = Report.from_runs(all_scores, output_path=out).save()
    # BLEU-vs-α curve over the held-out test set: sweep over the `alpha` column
    # we tagged onto each row. The single beam is inferred (no beam= needed).
    report.plot_sweep(
        "bleu", x="alpha",
        xlabel="α (merge weight)", ylabel_left="BLEU",
        title="BLEU vs merge weight α",
    )
    print(f"\nReport + plots saved to: {os.path.abspath(out)}\n")
    print(report)
    print(
        "\nRead the table left → right (α: 0 → 1). A flat or concave curve\n"
        "means A and B landed in the same basin; a U-shape means they didn't\n"
        "and the average crosses a high-loss ridge."
    )


if __name__ == "__main__":
    main()
