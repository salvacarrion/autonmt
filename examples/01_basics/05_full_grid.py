"""
============================================================================
 Tutorial 05 — The full grid, plus eval_mode
============================================================================

What you'll learn
-----------------
- How to sweep multiple axes at once (training size × subword model × vocab size).
- What the `sizes` field does: variants of the same corpus truncated to N lines.
- What `eval_mode` does inside `PredictConfig`:
    * "same"       — evaluate ONLY on test sets with the same (name, lang, size)
                     as the trained dataset.
    * "compatible" — evaluate on every test set with the same language pair.
    * "all"        — evaluate on everything.

The "compatible" mode is the most useful one for cross-corpus generalisation:
train on small subsets, evaluate on the full test sets of every compatible
corpus you have.

What's new vs tutorial 04
-------------------------
- The `encoding` block has TWO entries — once you list multiple entries, you
  can mix subword models freely (each entry expands its own vocab_sizes list).
- The `sizes` field adds a truncated variant, doubling the cells.
- `eval_mode` is set explicitly and explained in context.

The cross-product below is 2 sizes × (2 BPE vocabs + 1 char vocab) = 6 cells.
Training all six on a laptop is fine but takes a while; reduce `max_epochs`
or shrink the lists below to iterate faster.

Run
---
    pip install -e '.[hf]'
    python examples/01_basics/05_full_grid.py
"""
import datetime
import os

from tokenizers.normalizers import NFKC, Strip

from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.nn.models import Transformer
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.datasets.preprocessing import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.reporting.report import Report

BASE_PATH = "datasets/05_full_grid"
DATASET = "multi30k"
LANG_PAIR = "de-en"


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

    # The grid:
    #   sizes:     [("original", None), ("10k", 10_000)]                 (2 entries)
    #   encoding:  [{bpe × [2000, 4000]},                                 (2 entries)
    #               {char × [200]}]                                       (1 entry)
    #   ⇒ 2 × (2 + 1) = 6 dataset cells.
    #
    # The size label is part of the dataset identity, so the "10k" cell lives
    # at a *different* path on disk to the "original" cell — no risk of one
    # overwriting the other. The reserved name "original" is the un-truncated
    # reference and ignores any line cap.
    builder = DatasetBuilder(
        base_path=BASE_PATH,
        datasets=[{
            "name": DATASET,
            "languages": [LANG_PAIR],
            "sizes": [("original", None), ("10k", 10_000)],
        }],
        encoding=[
            {"subword_models": ["bpe"], "vocab_sizes": [2000, 4000]},
            {"subword_models": ["char"], "vocab_sizes": [200]},
        ],
        preprocess_raw_fn=preprocess_train,
        preprocess_splits_fn=preprocess_train,
        merge_vocabs=False,
    ).build(force_overwrite=False)

    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    print(f"\n[grid] {len(tr_datasets)} variant(s) will be trained:")
    for ds in tr_datasets:
        print(f"   - {ds.variant_id(as_path=True)}")
    print()

    fit_cfg = FitConfig(max_epochs=2, batch_size=128, learning_rate=1e-3, seed=42)

    # eval_mode="compatible" — score every cell on every test set whose
    # language pair matches the trained model. With our grid, that means each
    # of the 6 trained models is evaluated on the 2 test sets (one per size
    # variant), so the final report has 12 rows.
    #
    # Switch to "same" to score only on the cell's own test set (6 rows).
    pred_cfg = PredictConfig(
        metrics={"bleu", "chrf"}, beams=[5],
        load_checkpoint="best",
        preprocess_fn=preprocess_predict,
        eval_mode="compatible",
    )

    scores = []
    for train_ds in tr_datasets:
        src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)
        model = Transformer.from_vocabs(src_vocab, tgt_vocab)

        trainer = AutonmtTranslator.from_dataset(
            train_ds, model=model,
            src_vocab=src_vocab, tgt_vocab=tgt_vocab,
            run_prefix="fullgrid",
        )
        trainer.fit(train_ds, config=fit_cfg)
        scores.append(trainer.predict(ts_datasets, config=pred_cfg))

    out = f".outputs/05_full_grid/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    report = Report.from_runs(scores, output_path=out).save()
    report.plot_comparison(
        "bleu", beam=5,
        xlabel="Train variant", ylabel="BLEU",
        title="Full grid: size × subword × vocab",
    )

    print(f"\nReport + plots saved to: {os.path.abspath(out)}\n")
    print(report)


if __name__ == "__main__":
    main()
