import autonmt as al
from autonmt import DatasetBuilder
from autonmt.tasks.translation.bundle.report import generate_report


def main(fairseq_args):
    # Create datasets for training
    tr_datasets = DatasetBuilder(
        base_path="/home/salva/datasets/",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
        ],
        subword_models=["char+bytes"],
        vocab_sizes=[8000],
        merge_vocabs=True,
        force_overwrite=False,
        interactive=True,
        use_cmd=False,
        conda_env_name=None,
    ).build(make_plots=False, safe=True)

    # Create datasets for testing
    ts_datasets = tr_datasets

    # Train & Score a model for each dataset
    scores = []
    for ds in tr_datasets:
        model = al.FairseqTranslator(model_ds=ds, safe_seconds=2,
                                     force_overwrite=True, interactive=False,
                                     use_cmd=False,
                                     conda_env_name="mltests",
                                     conda_fairseq_env_name="fairseq")  # Conda envs will soon be deprecated
        model.fit(max_epochs=5, learning_rate=0.001, criterion="cross_entropy", optimizer="adam", clip_norm=1.0,
                  update_freq=1, max_tokens=None, batch_size=64, patience=10, seed=1234, num_gpus=1,
                  fairseq_args=fairseq_args)
        eval_scores = model.predict(ts_datasets, metrics={"bleu", "chrf", "ter"}, beams=[1])
        scores.append(eval_scores)
    print(scores[0][0]['beams']['beam1'])

    # Make report
    # generate_report(scores=scores, metric_id="beam_1__sacrebleu_bleu", output_path=".outputs/fairseq", save_figures=True, show_figures=False)


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
    ]

    # Run grid
    main(fairseq_args=fairseq_cmd_args)
