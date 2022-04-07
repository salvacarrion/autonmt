import datetime

from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Strip, Lowercase

from autonmt.bundle.report import generate_report
from autonmt.preprocessing import DatasetBuilder
from autonmt.toolkits.fairseq import FairseqTranslator


def main(fairseq_args, fairseq_venv_path):
    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            {"name": "europarl", "languages": ["es-en", "fr-en", "de-en"], "sizes": [("original", None), ("100k", 100000)]},
            {"name": "scielo/health", "languages": ["es-en"], "sizes": [("100k", 100000)], "split_sizes": (None, 1000, 1000)},
        ],
        encoding=[
            {"subword_models": ["bpe", "unigram+bytes"], "vocab_sizes": [8000, 16000, 32000]},
            {"subword_models": ["bytes", "char", "char+bytes"], "vocab_sizes": [1000]},
        ],
        normalizer=normalizers.Sequence([NFKC(), Strip(), Lowercase()]),
        merge_vocabs=False,
        eval_mode="compatible",
    ).build(make_plots=False, force_overwrite=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # Train & Score a model for each dataset
    scores = []
    for ds in tr_datasets:
        wandb_params = None  #dict(project="fairseq", entity="salvacarrion")
        model = FairseqTranslator(model_ds=ds, wandb_params=wandb_params, force_overwrite=True, fairseq_venv_path=fairseq_venv_path)
        model.fit(max_epochs=1, batch_size=128, seed=1234, patience=10, num_workers=12, fairseq_args=fairseq_args)
        m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[1])
        scores.append(m_scores)

    # Make report and print it
    output_path = f".outputs/fairseq/{str(datetime.datetime.now())}"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="beam1__sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    # These args are pass to fairseq using our pipeline
    # Fairseq Command-line tools: https://fairseq.readthedocs.io/en/latest/command_line_tools.html
    fairseq_cmd_args = [
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

        "--no-epoch-checkpoints",
        "--maximize-best-checkpoint-metric",
        "--best-checkpoint-metric bleu",
        "--eval-bleu",
        "--eval-bleu-args '{\"beam\": 5}'",
        "--eval-bleu-print-samples",
        "--scoring sacrebleu",
        "--log-format simple",
        "--task translation",
        "--task translation",
    ]

    # Set venv path
    # To create new venvs: virtualenv -p $(which python) VENV_NAME
    fairseq_venv_path = "source /home/scarrion/.venvs/fairseq_venv/bin/activate"

    # Run grid
    main(fairseq_args=fairseq_cmd_args, fairseq_venv_path=fairseq_venv_path)

