import os
import autonmt as al
from autonmt import DatasetBuilder
from autonmt.tasks.translation.metrics import create_report


def main(fairseq_args):

    # Create datasets for training
    tr_datasets = DatasetBuilder(
        base_path="/home/salva/datasets",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
        ],
        subword_models=["word"],
        vocab_sizes=[8000],
    ).build(make_plots=True, safe=True)

    # Create datasets for testing
    ts_datasets = tr_datasets

    # Train & Score a model for each dataset
    scores = {}
    for train_ds in tr_datasets:
        model = al.FairseqTranslator(conda_fairseq_env_name="fairseq", conda_env_name="mltests", force_overwrite=False)
        model.fit(train_ds, fairseq_args=fairseq_args)
        m_scores = model.predict(ts_datasets, metrics={"bleu", "chrf", "ter"}, beams=[1, 5])
        scores[str(train_ds)] = m_scores

    # Make report
    create_report(metrics=scores, metric_id="beam_5__sacrebleu_bleu", output_path=".outputs", save_figures=True, show_figures=False)


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
        "--lr 0.001",
        "--optimizer adam",
        "--criterion cross_entropy",
        "--max-tokens 4096",
        "--max-epoch 10",
        "--clip-norm 1.0",
        "--update-freq 1",
        "--patience 10",
        "--seed 1234",
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
    main(fairseq_args=fairseq_cmd_args)
