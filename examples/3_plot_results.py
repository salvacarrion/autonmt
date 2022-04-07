import pandas as pd
from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Strip, Lowercase

from autonmt.bundle import utils
from autonmt.bundle.report import generate_multivariable_report
from autonmt.preprocessing import DatasetBuilder


def main(fairseq_args, fairseq_venv_path):

     # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
        ],
        encoding=[
            {"subword_models": ["unigram+bytes"], "vocab_sizes": [x+256 for x in [100, 200, 400, 1000, 2000, 4000, 8000, 16000]]},
        ],
        normalizer=normalizers.Sequence([NFKC(), Strip(), Lowercase()]),
        merge_vocabs=False,
        eval_mode="compatible",
    ).build(make_plots=False, force_overwrite=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # Train & Score a model for each dataset
    stats = []
    for ds in tr_datasets:
        # Get ds stats
        ds_stats = utils.load_json(ds.get_stats_path("stats.json"))

        # Add stats
        ds_stats["scores"] = {}
        row = {
            "subword_model": ds.subword_model,
            "vocab_size": ds.vocab_size,
            "unknown_avg_tokens": ds_stats["val.en"]["unknown_avg_tokens"],
        }
        stats.append(row)

    # Create dataframes
    # assert len(ts_datasets) == 1
    df_report = pd.DataFrame(stats)
    df_report["dataset"] = [f"{ds.dataset_name}-{ds.dataset_size_name}".replace("_lc", "").title() for ds in tr_datasets]
    df_report["vocab_size"] = df_report["vocab_size"].astype(int)

    # Make report and print it
    output_path = f".outputs/myplots"
    prefix = "unknowns_"
    generate_multivariable_report(data=df_report,
                           x="vocab_size",
                           y_left=("unknown_avg_tokens", "subword_model"), y_right=None,
                           output_path=output_path, prefix=prefix,
                           save_figures=True, show_figures=False, save_csv=True)
    print("Summary:")
    print(df_report.to_string(index=False))


if __name__ == "__main__":
    main()
