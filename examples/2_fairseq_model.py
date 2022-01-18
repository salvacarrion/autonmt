from autonmt.preprocessing import DatasetBuilder
from autonmt.bundle.report import generate_report

from autonmt.toolkits.fairseq import FairseqTranslator

import os
import datetime


def main(fairseq_args, fairseq_venv_path):
    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            # {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
        ],
        subword_models=["unigram"],
        vocab_sizes=[4000],
        merge_vocabs=False,
        force_overwrite=False,
        use_cmd=True,
        eval_mode="same",
        letter_case="lower",
        venv_path="source /home/scarrion/.venvs/mltests_venv/bin/activate",
    ).build(make_plots=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_ds()
    ts_datasets = builder.get_ds(ignore_variants=True)

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

