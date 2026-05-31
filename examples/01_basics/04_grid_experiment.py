"""
============================================================================
 Tutorial 04 — Your first grid experiment
============================================================================

What you'll learn
-----------------
This is the feature AutoNMT was built around: declaring a *grid* of variants
and training one model per cell, then producing a single comparative report.

Here we sweep one axis — vocab size — so the question being answered is:
    "How does BLEU change when I shrink/grow the BPE vocabulary?"

The loop pattern shown here generalises: every axis you add to `datasets=` or
`encoding=` multiplies the number of cells, and the for-loop downstream is the
same. Tutorial 05 adds more axes; tutorial 06 swaps the backend.

What's new vs tutorial 03
-------------------------
- The `encoding` block lists *multiple* `vocab_sizes`, expanding the
  cross-product to N dataset variants.
- We loop over `builder.get_train_ds()` and append per-cell scores into one
  list, then hand that list to `Report.from_runs(...)` for a side-by-side report.
- `report.plot_comparison(...)` writes a grouped bar chart of BLEU per cell.

Run
---
    pip install -e '.[hf]'
    python examples/01_basics/04_grid_experiment.py
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

BASE_PATH = "datasets/04_grid"
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

    # Declare the grid. With one dataset × one lang pair × one size × one
    # subword model × three vocab sizes, the builder unrolls to 3 cells.
    builder = DatasetBuilder(
        base_path=BASE_PATH,
        datasets=[{
            "name": DATASET,
            "languages": [LANG_PAIR],
            "sizes": [("original", None)],
        }],
        encoding=[{
            "subword_models": ["bpe"],
            "vocab_sizes": [2000, 4000, 8000],   # ← the swept axis
        }],
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
    pred_cfg = PredictConfig(
        metrics={"bleu", "chrf"}, beams=[1, 5],
        load_checkpoint="best",
        preprocess_fn=preprocess_predict,
        eval_mode="compatible",
    )

    # One model per cell. Each iteration writes its own checkpoints/logs/eval
    # tree thanks to `from_dataset`, so re-running the script picks up where
    # the previous run left off (cached stages are skipped).
    scores = []
    for train_ds in tr_datasets:
        src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)
        model = Transformer.from_vocabs(src_vocab, tgt_vocab)

        trainer = AutonmtTranslator.from_dataset(
            train_ds, model=model,
            src_vocab=src_vocab, tgt_vocab=tgt_vocab,
            run_prefix="grid",
        )
        trainer.fit(train_ds, config=fit_cfg)
        scores.append(trainer.predict(ts_datasets, config=pred_cfg))

    # Report. `df_report` is the wide DataFrame; `df_summary` is a small subset
    # ready to print. Both are also saved as CSV under `reports/`.
    out = f".outputs/04_grid/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    report = Report.from_runs(scores, output_path=out).save()

    # Bar chart of BLEU per cell. Ask for the metric by short name — `Report`
    # resolves it to the right `translations.beam<N>.<tool>_<metric>_<field>`
    # column (here, two beams exist, so we pin beam=5).
    report.plot_comparison(
        "bleu", beam=5,
        xlabel="Vocab size variant", ylabel="BLEU",
        title=f"Vocab-size sweep on {DATASET} ({LANG_PAIR})",
    )

    print(f"\nReport + plots saved to: {os.path.abspath(out)}\n")
    print(report)


if __name__ == "__main__":
    main()
