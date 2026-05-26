"""Grid experiment with the built-in Transformer on multiple datasets / vocabs.

Sweeps {europarl × {es-en, fr-en, de-en} × {original, 100k}} × subword grids,
trains one model per cell and emits a single report comparing them.

``byte_fallback=True`` enables SentencePiece's byte fallback (rare-character
unknowns are emitted as bytes instead of <unk>); it's orthogonal to the
subword model. Two equivalent forms:

    {"subword_models": ["bpe"], "vocab_sizes": [...], "byte_fallback": True}
    {"subword_models": ["bpe+bytes"], "vocab_sizes": [...]}             # sugar

To compare a model with/without fallback, declare two encoding entries.

Expects the raw parallel corpora under ``datasets/translate/<name>/<lang>/<size>/data/0_raw/``.
See README ▸ "On-disk layout" for the exact expected structure.
"""
import datetime
import os

from tokenizers.normalizers import NFKC, Strip

from autonmt.reporting.figures import plot_model_comparison
from autonmt.reporting.report import generate_report
from autonmt.core.models import Transformer
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.processors import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.backends import AutonmtTranslator
from autonmt.backends.base.config import FitConfig, PredictConfig
from autonmt.vocabularies import Vocabulary


def normalize(x):
    return normalize_lines(x, seq=[NFKC(), Strip()])


def preprocess_raw(data, ds):
    return preprocess_pairs(
        data["src"]["lines"], data["trg"]["lines"],
        normalize_fn=normalize, min_len=1, remove_duplicates=False, shuffle_lines=True,
    )


def preprocess_splits(data, ds):
    return preprocess_pairs(
        data["src"]["lines"], data["trg"]["lines"], normalize_fn=normalize,
    )


def preprocess_predict(data, ds):
    return preprocess_lines(data["lines"], normalize_fn=normalize)


def main():
    builder = DatasetBuilder(
        base_path="datasets/translate",
        datasets=[
            {"name": "europarl", "languages": ["es-en", "fr-en", "de-en"],
             "sizes": [("original", None), ("100k", 100_000)]},
            {"name": "scielo/health", "languages": ["es-en"],
             "sizes": [("100k", 100_000)], "split_sizes": (None, 1000, 1000)},
        ],
        encoding=[
            # Canonical form
            {"subword_models": ["bpe"], "vocab_sizes": [8000, 16000, 32000]},
            {"subword_models": ["unigram"], "vocab_sizes": [8000, 16000, 32000], "byte_fallback": True},
            # Sugar form: "<model>+bytes" expands to byte_fallback=True for that model
            {"subword_models": ["bytes", "char", "char+bytes"], "vocab_sizes": [1000]},
        ],
        preprocess_raw_fn=preprocess_raw,
        preprocess_splits_fn=preprocess_splits,
        merge_vocabs=False,
    ).build(force_overwrite=False)

    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # One model per (dataset × size × subword × vocab_size) cell.
    fit_cfg = FitConfig(
        max_epochs=5, batch_size=128, learning_rate=1e-3, optimizer="adam",
        patience=10, num_workers=10, seed=1234,
    )
    pred_cfg = PredictConfig(
        metrics={"bleu", "chrf", "bertscore"}, beams=[1, 5],
        load_checkpoint="best", preprocess_fn=preprocess_predict,
        eval_mode="compatible",
    )

    scores = []
    for train_ds in tr_datasets:
        src_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.src_lang)
        trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.trg_lang)
        model = Transformer(
            src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab),
            padding_idx=src_vocab.pad_id,
        )

        trainer = AutonmtTranslator(
            model=model, src_vocab=src_vocab, trg_vocab=trg_vocab,
            runs_dir=train_ds.get_runs_path(toolkit="autonmt"),
            run_name=train_ds.get_run_name(run_prefix="mymodel"),
        )

        # `strategy` and the loggers (wandb/comet) are not part of FitConfig — they
        # are toolkit-specific extras and pass through `**kwargs` untouched.
        trainer.fit(train_ds, config=fit_cfg, strategy="ddp", save_best=True, save_last=True)
        scores.append(trainer.predict(ts_datasets, config=pred_cfg))

    # Report
    output_path = f".outputs/autonmt/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path)
    print("\nSummary:")
    print(df_summary.to_string(index=False))

    plot_model_comparison(
        out_dir=os.path.join(output_path, "plots"),
        df_report=df_report,
        metric="translations.beam1.sacrebleu_bleu_score",
        xlabel="MT Models", ylabel="BLEU", title="Model comparison",
    )


if __name__ == "__main__":
    main()
