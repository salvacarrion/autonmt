import pandas as pd

from autonmt.bundle import utils
from autonmt.bundle.report import generate_multivariable_report
from autonmt.preprocessing import DatasetBuilder

from autonmt.preprocessing.processors import preprocess_pairs, preprocess_lines, normalize_lines

# Preprocess functions
normalize_fn = lambda x: normalize_lines(x)
preprocess_raw_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn, min_len=1, max_len=None, remove_duplicates=True, shuffle_lines=True)
preprocess_splits_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn)
preprocess_predict_fn = lambda x: preprocess_lines(x, normalize_fn=normalize_fn)

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
