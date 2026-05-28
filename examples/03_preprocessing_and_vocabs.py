"""
============================================================================
 Tutorial 03 — Preprocessing & subword vocabularies
============================================================================

What you'll learn
-----------------
- What `preprocess_raw_fn` and `preprocess_splits_fn` actually do.
- The filters offered by `preprocess_pairs` (length bounds, dedupe, length-ratio).
- How to choose a subword model: `bpe`, `unigram`, `word`, `char`, `bytes`.
- The `<model>+bytes` sugar that enables SPM's `byte_fallback`.

Where preprocessing runs in the pipeline
----------------------------------------
    raw files  ──preprocess_raw_fn──►  splits  ──preprocess_splits_fn──►  encoded
                                          │
                                          └─ at predict() time, the test source
                                             is run through preprocess_fn (PredictConfig)
                                             before being encoded.

Use `preprocess_raw_fn` for opinionated decisions you only want applied ONCE
to the corpus (e.g. dedupe + shuffle the raw lines). Use `preprocess_splits_fn`
for normalization you want applied to each split *independently* so val/test
get the exact same treatment as train.

What's new vs tutorial 02
-------------------------
We bring the dataset down from `multi30k` (so the script runs without your own
files), apply non-trivial preprocessing, and switch the subword model to
`unigram+bytes` to show byte-fallback in action.

Run
---
    pip install -e '.[hf]'
    python examples/03_preprocessing_and_vocabs.py
"""
import datetime
import os

from tokenizers.normalizers import NFKC, Lowercase, Strip

from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.models import Transformer
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.datasets.processors import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.reporting.report import format_summary_table, generate_report

BASE_PATH = "datasets/03_preproc"
DATASET = "multi30k"
LANG_PAIR = "de-en"


def normalize(lines):
    # NFKC compat-fold, strip whitespace, lowercase.
    # Swap or extend this list with any HF `tokenizers.normalizers.*`.
    return normalize_lines(lines, seq=[NFKC(), Strip(), Lowercase()])


def preprocess_raw(data, ds):
    # Aggressive cleaning we ONLY want applied to the raw corpus, once.
    return preprocess_pairs(
        data["src"]["lines"], data["trg"]["lines"],
        normalize_fn=normalize,
        min_len=1,                       # drop empty lines
        max_len_percentile=99,           # drop the longest 1% (likely noise)
        remove_duplicates=True,          # drop exact-duplicate pairs
        max_len_ratio_percentile=99,     # drop pairs whose src/trg length ratio is extreme
        shuffle_lines=True,              # shuffle once so the order of the raw file doesn't bias splits
    )


def preprocess_splits(data, ds):
    # Per-split normalization. Same treatment for train/val/test — DO NOT dedupe
    # or shuffle here, those decisions belong to `preprocess_raw_fn`.
    return preprocess_pairs(data["src"]["lines"], data["trg"]["lines"], normalize_fn=normalize)


def preprocess_predict(data, ds):
    # Test source seen at predict() time. Keep this identical to what was done
    # at training time (modulo per-pair filters which obviously can't apply here).
    return preprocess_lines(data["lines"], normalize_fn=normalize)


def main():
    download_hf_dataset(
        hf_id="bentrevett/multi30k", base_path=BASE_PATH,
        dataset_name=DATASET, lang_pair=LANG_PAIR,
        src_field="de", trg_field="en",
    )

    builder = DatasetBuilder(
        base_path=BASE_PATH,
        datasets=[{
            "name": DATASET,
            "languages": [LANG_PAIR],
            "sizes": [("original", None)],
        }],
        # `unigram+bytes` is sugar for `{"subword_models": ["unigram"], "byte_fallback": True}`.
        # Byte-fallback emits unseen characters as raw bytes instead of <unk>, which
        # is invaluable when the corpus has rare scripts, emoji, or names.
        #
        # Alternatives you can try by editing this line:
        #   "bpe"          — classic BPE merges
        #   "word"         — whitespace + Moses pretokenization (slow vocab, fast model)
        #   "char"         — one symbol per character
        #   "bytes"        — pure byte-level (vocab=256, ignores vocab_sizes)
        encoding=[{"subword_models": ["unigram+bytes"], "vocab_sizes": [4000]}],
        preprocess_raw_fn=preprocess_raw,
        preprocess_splits_fn=preprocess_splits,
        merge_vocabs=False,
    ).build(force_overwrite=False, verbose=True)  # verbose=True prints per-split stats

    train_ds = builder.get_train_ds()[0]
    test_datasets = builder.get_test_ds()

    # The trained SPM model and the tab-separated frequency vocab end up under
    # `<BASE_PATH>/<dataset>/<lang>/<size>/vocabs/`. Inspect them after the
    # build to debug your subword choice.
    print(f"\n[info] Vocab artifacts at: {train_ds.get_vocab_path()}")
    print(f"[info] Encoded splits at:   {train_ds.get_encoded_path()}\n")

    src_vocab, trg_vocab = train_ds.build_vocabs(max_tokens=150)
    model = Transformer.from_vocabs(src_vocab, trg_vocab)

    trainer = AutonmtTranslator.from_dataset(
        train_ds, model=model,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        run_prefix="preproc",
    )

    trainer.fit(train_ds, config=FitConfig(max_epochs=2, batch_size=128, learning_rate=1e-3, seed=42))
    scores = trainer.predict(
        test_datasets,
        config=PredictConfig(
            metrics={"bleu", "chrf"}, beams=[5],
            load_checkpoint="best",
            preprocess_fn=preprocess_predict,
            eval_mode="compatible",
        ),
    )

    out = f".outputs/03_preproc/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    _, df_summary = generate_report(scores=[scores], output_path=out)
    print(f"\nReport saved to: {os.path.abspath(out)}\n")
    print(format_summary_table(df_summary))


if __name__ == "__main__":
    main()
