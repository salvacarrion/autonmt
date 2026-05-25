"""Same grid as example #1, but using the Fairseq CLI as the backend.

⚠️  DEPRECATED EXAMPLE  ⚠️
    Fairseq was archived by Meta on 2026-03-20 and is no longer maintained
    (https://github.com/facebookresearch/fairseq). This example is kept for
    users with an existing fairseq install; new projects should use
    ``examples/1_custom_model.py`` (AutonmtTranslator + PyTorch Lightning).

Demonstrates AutoNMT's toolkit abstraction: only the trainer class changes
(``AutonmtTranslator`` → ``FairseqTranslator``). Subword models, dataset
layout, scoring and reports are identical.

Fairseq is an *optional* dependency and is NOT in ``requirements.txt``.
Install it manually (``pip install fairseq``) before running this example;
``FairseqTranslator()`` will otherwise raise ``ImportError``.
"""
import datetime
import os

from tokenizers.normalizers import NFKC, Strip

from autonmt.reporting.figures import plot_model_comparison
from autonmt.reporting.report import generate_report
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.processors import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.backends.base.config import FitConfig, PredictConfig
from autonmt.backends.fairseq.translator import FairseqTranslator


def normalize(x):
    return normalize_lines(x, seq=[NFKC(), Strip()])


def preprocess_predict(x):
    return preprocess_lines(x, normalize_fn=normalize)


# Fairseq CLI flags. Any flag here ALWAYS wins over the equivalent AutoNMT kwarg.
FAIRSEQ_MODEL_ARGS = [
    "--arch transformer",
    "--encoder-embed-dim 256",
    "--decoder-embed-dim 256",
    "--encoder-layers 3",
    "--decoder-layers 3",
    "--encoder-attention-heads 8",
    "--decoder-attention-heads 8",
    "--encoder-ffn-embed-dim 512",
    "--decoder-ffn-embed-dim 512",
    "--dropout 0.1",
]
FAIRSEQ_TRAINING_ARGS = [
    "--no-epoch-checkpoints",
    "--maximize-best-checkpoint-metric",
    "--best-checkpoint-metric bleu",
    "--eval-bleu",
    '--eval-bleu-args {"beam": 5}',
    "--eval-bleu-print-samples",
    "--scoring sacrebleu",
    "--log-format simple",
    "--task translation",
]


def main(fairseq_args):
    builder = DatasetBuilder(
        base_path="datasets/translate",
        datasets=[
            {"name": "europarl", "languages": ["es-en"], "sizes": [("50k", 50_000)]},
        ],
        encoding=[
            {"subword_models": ["word"], "vocab_sizes": [32000]},
            {"subword_models": ["bpe"], "vocab_sizes": [8000, 16000, 32000]},
            {"subword_models": ["bytes", "char"], "vocab_sizes": [1000]},
        ],
        preprocess_raw_fn=lambda x, y: preprocess_pairs(
            x, y, normalize_fn=normalize, min_len=1, remove_duplicates=True, shuffle_lines=True),
        preprocess_splits_fn=lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize),
        merge_vocabs=False,
    ).build(force_overwrite=False)

    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    fit_cfg = FitConfig(
        max_epochs=5, batch_size=128, learning_rate=1e-3, optimizer="adam",
        patience=10, num_workers=0, seed=1234,
    )
    pred_cfg = PredictConfig(
        metrics={"bleu"}, beams=[1], load_checkpoint="best",
        preprocess_fn=preprocess_predict, eval_mode="compatible",
    )

    scores = []
    for train_ds in tr_datasets:
        trainer = FairseqTranslator(
            runs_dir=train_ds.get_runs_path(toolkit="autonmt"),
            run_name=train_ds.get_run_name(run_prefix="mymodel"),
        )
        trainer.fit(train_ds, config=fit_cfg, strategy="ddp", fairseq_args=fairseq_args)
        scores.append(trainer.predict(ts_datasets, config=pred_cfg))

    output_path = f".outputs/fairseq/{datetime.datetime.now():%Y%m%d_%H%M%S}"
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
    main(fairseq_args=FAIRSEQ_MODEL_ARGS + FAIRSEQ_TRAINING_ARGS)
