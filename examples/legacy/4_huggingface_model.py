"""Evaluate a pretrained HuggingFace MT model on AutoNMT-prepared test sets.

Loads any seq2seq model from the Hub (e.g. ``Helsinki-NLP/opus-mt-de-en``)
and runs the same ``predict()`` pipeline as :mod:`examples.1_custom_model`,
so you can compare a custom-trained AutoNMT model against off-the-shelf
HuggingFace baselines on the same data.

Phase 1 supports **inference only** — fine-tuning HF models via ``.fit()``
is not implemented yet (will raise ``NotImplementedError``).

Run with:  pip install -e '.[hf,hf-models]' && python examples/4_huggingface_model.py
"""
import datetime
import os

from tokenizers.normalizers import NFKC, Strip

from autonmt.reporting.report import format_summary_table, generate_report
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.datasets.processors import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.backends import HuggingFaceTranslator
from autonmt.backends._base.config import PredictConfig

BASE_PATH = "datasets/quickstart"
DATASET = "multi30k"
LANG_PAIR = "de-en"


def normalize(x):
    return normalize_lines(x, seq=[NFKC(), Strip()])


def preprocess_train(data, ds):
    return preprocess_pairs(data["src"]["lines"], data["trg"]["lines"], normalize_fn=normalize)


def preprocess_predict(data, ds):
    return preprocess_lines(data["lines"], normalize_fn=normalize)


def main():
    # 1. Materialize the dataset (same layout the other examples use).
    download_hf_dataset(
        hf_id="bentrevett/multi30k", base_path=BASE_PATH,
        dataset_name=DATASET, lang_pair=LANG_PAIR,
        src_field="de", trg_field="en",
    )

    builder = DatasetBuilder(
        base_path=BASE_PATH,
        datasets=[{"name": DATASET, "languages": [LANG_PAIR], "sizes": [("original", None)]}],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [4000]}],
        preprocess_raw_fn=preprocess_train,
        preprocess_splits_fn=preprocess_train,
        merge_vocabs=False,
    ).build(force_overwrite=False)

    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # 2. Evaluate a pretrained HuggingFace model on every dataset cell.
    pred_cfg = PredictConfig(
        metrics={"bleu", "chrf"}, beams=[1],
        preprocess_fn=preprocess_predict,
        eval_mode="compatible",
        batch_size=8,
    )

    scores = []
    for train_ds in tr_datasets:
        # ``from_dataset`` reuses the dataset's runs path and auto-fills
        # src_lang / trg_lang from the variant — same shape as AutonmtTranslator.
        trainer = HuggingFaceTranslator.from_dataset(
            train_ds,
            model_id="Helsinki-NLP/opus-mt-de-en",
            run_prefix="opus-mt-baseline",
            device="auto",
        )

        # No .fit() — the model is already pretrained.
        scores.append(trainer.predict(ts_datasets, config=pred_cfg))

    # 3. Report alongside any other model you've already scored on these datasets.
    output_path = f".outputs/huggingface/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    _, df_summary = generate_report(scores=scores, output_path=output_path)
    print(f"\nReport saved to: {os.path.abspath(output_path)}\n")
    print(format_summary_table(df_summary))


if __name__ == "__main__":
    main()
