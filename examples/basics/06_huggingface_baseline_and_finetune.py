"""
============================================================================
 Tutorial 06 — Backend swap: HuggingFace baseline and fine-tuning
============================================================================

What you'll learn
-----------------
The Translator API is backend-agnostic. The same `fit() → predict()` flow
you've used so far works with HuggingFace seq2seq checkpoints via
`HuggingFaceTranslator`, so you can:

    1. Evaluate a *pretrained* HF model on YOUR test set without training it.
    2. Fine-tune that same HF model on YOUR training set.
    3. Drop both runs into the same report alongside your custom AutoNMT model
       — they speak the same score schema.

What's new vs tutorial 05
-------------------------
- We import `HuggingFaceTranslator` instead of `AutonmtTranslator`. Everything
  else stays the same: `DatasetBuilder`, `FitConfig`, `PredictConfig`,
  `generate_report`.
- The HF backend brings its own tokenizer, so the `subword_models`/`vocab_sizes`
  declared in `encoding=` are only used for the AutoNMT-side dataset stats and
  cache layout; HF will ignore the SPM model and tokenize source text itself.
- `_translate` for HF writes `src.txt`/`ref.txt`/`hyp.txt` directly (direct
  mode); no SPM round-trip happens. From the user's POV, nothing changes.

Run
---
    pip install -e '.[hf,hf-models]'
    python examples/basics/06_huggingface_baseline_and_finetune.py
"""
import datetime
import os

from tokenizers.normalizers import NFKC, Strip

from autonmt.backends import HuggingFaceTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.datasets.preprocessing import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.reporting.report import format_summary_table, generate_report

BASE_PATH = "datasets/06_hf"
DATASET = "multi30k"
LANG_PAIR = "de-en"

# A small, lang-pair-specific Helsinki MARIAN checkpoint. Works out of the box
# for de→en and downloads in seconds. Pick any seq2seq id from the Hub here.
HF_MODEL = "Helsinki-NLP/opus-mt-de-en"


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

    # Even though HF brings its own tokenizer, the DatasetBuilder still owns
    # the on-disk layout, stats, and the AutoNMT-side eval flow — so we build
    # exactly like in the previous tutorials. The subword model here is just
    # bookkeeping for the dataset's identity.
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

    pred_cfg = PredictConfig(
        metrics={"bleu", "chrf"}, beams=[5],
        preprocess_fn=preprocess_predict,
        eval_mode="compatible",
        batch_size=8,
    )

    # --- (a) Pretrained baseline ----------------------------------------
    # No `.fit()` — the model is already trained. We jump straight to
    # `.predict()` and the report will include the HF model's BLEU/chrF
    # alongside anything else you've scored on these datasets.
    baseline = HuggingFaceTranslator.from_dataset(
        train_ds,
        model_id=HF_MODEL,
        run_prefix="opus-baseline",
        device="auto",
    )
    baseline_scores = baseline.predict(test_datasets, config=pred_cfg)

    # --- (b) Fine-tune the same checkpoint ------------------------------
    # FitConfig fields map onto `transformers.Seq2SeqTrainingArguments`:
    #   max_epochs → num_train_epochs
    #   batch_size → per_device_{train,eval}_batch_size
    #   learning_rate, weight_decay, gradient_clip_val, accumulate_grad_batches,
    #   patience (→ EarlyStoppingCallback), seed, num_workers.
    # HF-only knobs (label smoothing, mixed precision, ...) go through
    # `hf_training_args=dict(...)` on the .fit() call and win on collision.
    finetuner = HuggingFaceTranslator.from_dataset(
        train_ds,
        model_id=HF_MODEL,
        run_prefix="opus-finetuned",
        device="auto",
    )
    finetuner.fit(
        train_ds,
        config=FitConfig(
            max_epochs=1, batch_size=8, learning_rate=2e-5,
            weight_decay=0.01, gradient_clip_val=1.0,
            patience=2, num_workers=0, seed=42,
            save_best=True, monitor="eval_loss",
        ),
        hf_training_args={"label_smoothing_factor": 0.1},
    )
    finetuned_scores = finetuner.predict(test_datasets, config=pred_cfg)

    # Both runs land in the same report — one row per (run, test_ds) pair.
    out = f".outputs/06_hf/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    _, df_summary = generate_report(
        scores=[baseline_scores, finetuned_scores], output_path=out,
    )
    print(f"\nReport saved to: {os.path.abspath(out)}\n")
    print(format_summary_table(df_summary))


if __name__ == "__main__":
    main()
