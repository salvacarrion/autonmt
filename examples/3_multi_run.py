from autonmt.preprocessing import DatasetBuilder
from autonmt.bundle.report import generate_report

from autonmt.toolkits.fairseq import FairseqTranslator
import datetime


def main(fairseq_args):
    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/salva/datasets/",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
        ],
        subword_models=["word"],
        vocab_sizes=[8000],
        merge_vocabs=True,
        force_overwrite=False,
    ).build(make_plots=False, safe=True)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_ds()
    ts_datasets = builder.get_ds(ignore_variants=True)

    # Train & Score a model for each dataset
    runs = 3
    scores = []
    for ds in tr_datasets:

        run_eval_scores = []
        for i in range(runs):
            run_id = f"model{i}"
            model = FairseqTranslator(model_ds=ds, run_prefix=run_id, force_overwrite=False, conda_fairseq_env_name="fairseq")
            model.fit(max_epochs=10, batch_size=128, seed=1234, num_workers=16, fairseq_args=fairseq_args)
            m_scores = model.predict(ts_datasets, metrics={"bleu", "chrf", "ter"}, beams=[1, 5])
            run_eval_scores.append(m_scores)
        scores.append(run_eval_scores)

    # Make report and print it
    output_path = f".outputs/fairseq_multirun/{str(datetime.datetime.now())}"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="beam1__sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    # These args are pass to fairseq using our pipeline
    # Fairseq Command-line tools: https://fairseq.readthedocs.io/en/latest/command_line_tools.html
    fairseq_args = [
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
    ]

    # Run grid
    main(fairseq_args=fairseq_args)
