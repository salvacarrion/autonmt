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
            # {"name": "multi30k_test", "languages": ["de-en"], "sizes": [("original", None)]},
            {"name": "europarl_lc", "languages": ["de-en"], "sizes": [("50k", 50000)]},
        ],
        subword_models=["unigram"],
        vocab_sizes=[x for x in [8000, 16000]],
        merge_vocabs=False,
        force_overwrite=False,
        use_cmd=True,
        eval_mode="same",
        letter_case="lower",
        venv_path="source /home/scarrion/venvs/mltests_venv/bin/activate",
    ).build(make_plots=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_ds()
    ts_datasets = builder.get_ds(ignore_variants=True)

    # Train & Score a model for each dataset
    scores = []
    run_prefix = "transformer256emb"
    for ds in tr_datasets:
        try:
            wandb_params = dict(project="fairseq", entity="salvacarrion")
            model = FairseqTranslator(fairseq_venv_path=fairseq_venv_path,
                                      model_ds=ds, wandb_params=wandb_params, force_overwrite=True, run_prefix=run_prefix)
            model.fit(max_epochs=100, max_tokens=4096, batch_size=None, seed=1234, patience=10, num_workers=12, devices=1, fairseq_args=fairseq_args)
            m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[5])
            scores.append(m_scores)
        except Exception as e:
            print(e)

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

        "--lr 0.0005",
        "--criterion label_smoothed_cross_entropy --label-smoothing 0.1",
        "--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000",
        "--clip-norm 0.0",
    ]

    # Set venv path
    # To create new venvs: virtualenv -p $(which python) VENV_NAME
    fairseq_venv_path = "source /home/scarrion/venvs/fairseq_venv/bin/activate"

    # Run grid
    main(fairseq_args=fairseq_cmd_args, fairseq_venv_path=fairseq_venv_path)
