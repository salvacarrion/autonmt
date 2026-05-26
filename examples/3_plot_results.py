import pandas as pd

from autonmt.utils import fileio as utils
from autonmt.reporting.report import generate_sweep_report
from autonmt.datasets import DatasetBuilder

from autonmt.datasets.processors import preprocess_pairs, normalize_lines


def normalize_fn(x):
    return normalize_lines(x)


def preprocess_raw_fn(data, ds):
    return preprocess_pairs(
        data["src"]["lines"], data["trg"]["lines"],
        normalize_fn=normalize_fn, min_len=1, max_len=None,
        remove_duplicates=True, shuffle_lines=True,
    )


def preprocess_splits_fn(data, ds):
    return preprocess_pairs(
        data["src"]["lines"], data["trg"]["lines"], normalize_fn=normalize_fn,
    )


def main():
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
    ).build(force_overwrite=False)

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
    generate_sweep_report(data=df_report,
                          x="vocab_size",
                          y_left=("unknown_avg_tokens", "subword_model"), y_right=None,
                          output_path=output_path, prefix=prefix, save_csv=True)
    print("Summary:")
    print(df_report.to_string(index=False))


if __name__ == "__main__":
    main()
