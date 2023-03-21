import datetime

from autonmt.bundle.report import generate_report
from autonmt.modules.models import Transformer
from autonmt.preprocessing import DatasetBuilder
from autonmt.toolkits import FairseqTranslator
from autonmt.vocabularies import Vocabulary

from autonmt.preprocessing.processors import preprocess_pairs, preprocess_lines, normalize_lines
from tokenizers.normalizers import NFKC, Strip

# Preprocess functions
normalize_fn = lambda x: normalize_lines(x, seq=[NFKC(), Strip()])
preprocess_raw_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn, min_len=1, max_len=None, remove_duplicates=True, shuffle_lines=True)
preprocess_splits_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn)
preprocess_predict_fn = lambda x: preprocess_lines(x, normalize_fn=normalize_fn)

def main(fairseq_args):
    # Create preprocessing for training
    # Create preprocessing for training
    builder = DatasetBuilder(
        # Root folder for datasets
        base_path="datasets/translate",

        # Set of datasets, languages, training sizes to try
        datasets=[
            {"name": "europarl", "languages": ["es-en", "fr-en", "de-en"], "sizes": [("original", None), ("100k", 100000)]},
            {"name": "scielo/health", "languages": ["es-en"], "sizes": [("100k", 100000)], "split_sizes": (None, 1000, 1000)},
        ],

        # Set of subword models and vocab sizes to try
        encoding=[
            {"subword_models": ["bpe", "unigram+bytes"], "vocab_sizes": [8000, 16000, 32000]},
            {"subword_models": ["bytes", "char", "char+bytes"], "vocab_sizes": [1000]},
        ],

        # Preprocessing functions
        preprocess_raw_fn=preprocess_raw_fn,
        preprocess_splits_fn=preprocess_splits_fn,

        # Additional args
        merge_vocabs=False,
    ).build(make_plots=False, force_overwrite=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # Train & Score a model for each dataset
    scores = []
    for train_ds in tr_datasets:
        # Define trainer
        runs_dir = train_ds.get_runs_path(toolkit="autonmt")
        run_name = train_ds.get_run_name(run_prefix="mymodel")
        trainer = FairseqTranslator(runs_dir=runs_dir, run_name=run_name)

        # Train model
        trainer.fit(train_ds, max_epochs=5, learning_rate=0.001, optimizer="adam", batch_size=128, seed=1234,
                    patience=10, num_workers=10, strategy="ddp", fairseq_args=fairseq_args)

        # Test model
        m_scores = trainer.predict(ts_datasets, metrics={"bleu", "chrf", "bertscore"}, beams=[1, 5], load_checkpoint="best",
                                   preprocess_fn=preprocess_predict_fn, eval_mode="compatible", force_overwrite=False)
        scores.append(m_scores)

    # Make report and print it
    output_path = f".outputs/fairseq/{str(datetime.datetime.now())}"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="translations.beam1.sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    # These args are pass to fairseq using our pipeline
    # Fairseq Command-line tools: https://fairseq.readthedocs.io/en/latest/command_line_tools.html
    fairseq_model_args = [
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

    fairseq_training_args = [
        "--no-epoch-checkpoints",
        "--maximize-best-checkpoint-metric",
        "--best-checkpoint-metric bleu",
        "--eval-bleu",
        '--eval-bleu-args {\"beam\": 5}',
        "--eval-bleu-print-samples",
        "--scoring sacrebleu",
        "--log-format simple",
        "--task translation",
    ]

    cmd_args = fairseq_model_args+fairseq_training_args

    # Run grid
    main(fairseq_args=cmd_args)

