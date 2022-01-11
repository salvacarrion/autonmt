import pandas as pd

from autonmt.preprocessing import DatasetBuilder
from autonmt.bundle.report import generate_report, generate_vocabs_report

from autonmt.toolkits.fairseq import FairseqTranslator
from autonmt.bundle import utils

import os
import datetime


def main(fairseq_args, fairseq_venv_path):

    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            # {"name": "multi30k_test", "languages": ["de-en"], "sizes": [("original", None)]},
            # {"name": "europarl", "languages": ["de-en"], "sizes": [("100k_lc", 100000), ("50k_lc", 50000)]},
            {"name": "ccaligned", "languages": ["ti-en"], "sizes": [("original_lc", None)]},

        ],
        subword_models=["unigram+bytes"],
        vocab_sizes=[x+256 for x in [400, 1000, 2000, 4000, 8000]],
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
    stats = []
    run_prefix = "transformer256emb"
    for ds in tr_datasets:
        # Get ds stats
        ds_stats = utils.load_json(ds.get_stats_path("stats.json"))

        # Get scores
        model = FairseqTranslator(fairseq_venv_path=fairseq_venv_path, model_ds=ds, force_overwrite=False, run_prefix=run_prefix)
        m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[5], truncate_at=1023)

        # Add stats
        ds_stats["scores"] = {}
        row = {
            "subword_model": ds.subword_model,
            "vocab_size": ds.vocab_size,
            "avg_tokens": ds_stats["train.en"]["avg_tokens"],
            "bleu": m_scores[0]['beams']['beam5']['sacrebleu_bleu_score'],
        }
        stats.append(row)

    # Create dataframes
    assert len(ts_datasets) == 1
    df_bleu = pd.DataFrame(stats)
    df_bleu["dataset"] = str(ts_datasets[0])
    df_avg_tokens = pd.DataFrame(stats)

    # Make report and print it
    output_path = f".outputs/fairseq"
    # df_bleu = pd.read_csv(os.path.join(output_path, "reports", "my_report_vocab_bleu.csv"))
    # df_avg_tokens = pd.read_csv(os.path.join(output_path, "reports", "my_report_vocab_avg_tokens.csv"))
    generate_vocabs_report(data_left=df_bleu, data_right=df_avg_tokens,
                                              y_left=("bleu", "dataset"), y_right=("avg_tokens", None),
                                              output_path=output_path, prefix=str(ts_datasets[0]),
                                              save_figures=True, show_figures=False)
    print("Summary:")
    print(df_bleu.to_string(index=False))


if __name__ == "__main__":
    # Set venv path
    # To create new venvs: virtualenv -p $(which python) VENV_NAME
    fairseq_venv_path = "source /home/scarrion/venvs/fairseq_venv/bin/activate"

    # Run grid
    main(fairseq_args=None, fairseq_venv_path=fairseq_venv_path)