"""Fine-tune a pretrained HuggingFace seq2seq model on an AutoNMT dataset.

Same dataset layout as the other examples — but instead of training from
scratch, we load a pretrained MT checkpoint (e.g. ``Helsinki-NLP/opus-mt-de-en``)
and fine-tune it for a few epochs on the prepared splits, then score with the
existing metric pipeline.

The fine-tuning loop is :class:`transformers.Seq2SeqTrainer`. FitConfig fields
are mapped onto :class:`Seq2SeqTrainingArguments` (epochs, batch size, lr,
weight decay, gradient clipping, accumulation steps, early-stopping patience,
seed, num_workers). HF-specific knobs that don't have an AutoNMT analogue go
through ``hf_training_args=dict(...)`` and win on collision.

Run with:  pip install -e '.[hf,hf-models]' && python examples/5_huggingface_finetune.py
"""
import datetime
import os

from tokenizers.normalizers import NFKC, Strip

from autonmt.reporting.report import format_summary_table, generate_report
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.datasets.processors import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.backends import HuggingFaceTranslator
from autonmt.backends.base.config import FitConfig, PredictConfig

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
    # 1. Materialize the dataset (same layout as the other examples).
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

    # 2. Fine-tune + evaluate one HF model per dataset variant.
    fit_cfg = FitConfig(
        max_epochs=1, batch_size=64, learning_rate=2e-5,
        weight_decay=0.01, gradient_clip_val=1.0,
        patience=2, num_workers=2, seed=42,
        save_best=True, monitor="eval_loss",
    )
    pred_cfg = PredictConfig(
        metrics={"bleu", "chrf"}, beams=[1],
        preprocess_fn=preprocess_predict,
        eval_mode="compatible",
        batch_size=8,
    )

    scores = []
    for train_ds in tr_datasets:
        trainer = HuggingFaceTranslator.from_dataset(
            train_ds,
            model_id="Helsinki-NLP/opus-mt-de-en",
            run_prefix="opus-mt-ft",
            device="auto",
        )

        # FitConfig is mapped to Seq2SeqTrainingArguments. Extra HF-only knobs
        # (e.g. mixed precision, label smoothing) can be passed via
        # ``hf_training_args=`` and win on collision with the mapping.
        trainer.fit(
            train_ds, config=fit_cfg,
            hf_training_args={"label_smoothing_factor": 0.1},
        )

        # After fit, model_id points at the local checkpoint dir; predict()
        # picks up the fine-tuned weights automatically.
        scores.append(trainer.predict(ts_datasets, config=pred_cfg))

    # 3. Report.
    output_path = f".outputs/huggingface_ft/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    _, df_summary = generate_report(scores=scores, output_path=output_path)
    print(f"\nReport saved to: {os.path.abspath(output_path)}\n")
    print(format_summary_table(df_summary))


if __name__ == "__main__":
    main()
