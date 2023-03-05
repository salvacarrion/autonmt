import datetime
import os

from autonmt.bundle.report import generate_report
from autonmt.modules.models import Transformer
from autonmt.preprocessing import DatasetBuilder
from autonmt.toolkits import AutonmtTranslator
from autonmt.vocabularies import Vocabulary

from autonmt.preprocessing.processors import preprocess_pairs, preprocess_lines, normalize_lines
from autonmt.bundle.utils import read_file_lines, shuffle_in_order, write_file_lines, make_dir

# Preprocess functions
normalize_fn = lambda x: normalize_lines(x)
preprocess_raw_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn, min_len=1, max_len_percentile=99.95, max_len_ratio_percentile=99.95, remove_duplicates=True, shuffle_lines=True)
preprocess_splits_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn)

# Constants
src_alias = "en"
trg_alias = "xx"
DATASET = "europarl"
BASE_PATH = "datasets/translate"

# Create folders
output_path = os.path.join(BASE_PATH, f"{DATASET}/{src_alias}-{trg_alias}/original/data/1_splits")
make_dir([output_path])

def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        # Root folder for datasets
        base_path=BASE_PATH,

        # Set of datasets, languages, training sizes to try
        datasets=[
            {"name": DATASET, "languages": ["cs-en", "de-en", "el-en", "es-en", "fr-en", "it-en"], "sizes": [("100k", 100000)], "split_sizes": (None, 3000, 3000)},
        ],

        # Preprocessing functions
        preprocess_raw_fn=preprocess_raw_fn,
        preprocess_splits_fn=preprocess_splits_fn,
    ).build(make_plots=False, force_overwrite=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    print("Tagging lines with languages...")
    tr_lines_xx, tr_lines_yy = [], []
    vl_lines_xx, vl_lines_yy = [], []
    ts_lines_xx, ts_lines_yy = [], []
    for train_ds in tr_datasets:
        src_lang = train_ds.src_lang
        tgt_lang = train_ds.trg_lang

        tr_lines_src_tmp = read_file_lines(filename=train_ds.get_split_path(f"{'train'}.{src_lang}"), autoclean=False)
        tr_lines_tgt_tmp = read_file_lines(filename=train_ds.get_split_path(f"{'train'}.{tgt_lang}"), autoclean=False)

        if src_alias == "xx":
            tr_lines_xx += [f"<{src_lang}>-<{tgt_lang}>|{line}" for line in tr_lines_src_tmp]
            tr_lines_yy += tr_lines_tgt_tmp
        else:
            tr_lines_xx += [f"<{tgt_lang}>-<{src_lang}>|{line}" for line in tr_lines_tgt_tmp]
            tr_lines_yy += tr_lines_src_tmp

        vl_lines_src_tmp = read_file_lines(filename=train_ds.get_split_path(f"{'val'}.{src_lang}"), autoclean=False)
        vl_lines_tgt_tmp = read_file_lines(filename=train_ds.get_split_path(f"{'val'}.{tgt_lang}"), autoclean=False)
        if src_alias == "xx":
            vl_lines_xx += [f"<{src_lang}>-<{tgt_lang}>|{line}" for line in vl_lines_src_tmp]
            vl_lines_yy += vl_lines_tgt_tmp
        else:
            vl_lines_xx += [f"<{tgt_lang}>-<{src_lang}>|{line}" for line in vl_lines_tgt_tmp]
            vl_lines_yy += vl_lines_src_tmp

        ts_lines_src_tmp = read_file_lines(filename=train_ds.get_split_path(f"{'test'}.{src_lang}"), autoclean=False)
        ts_lines_tgt_tmp = read_file_lines(filename=train_ds.get_split_path(f"{'test'}.{tgt_lang}"), autoclean=False)
        if src_alias == "xx":
            ts_lines_xx += [f"<{src_lang}>-<{tgt_lang}>|{line}" for line in ts_lines_src_tmp]
            ts_lines_yy += ts_lines_tgt_tmp
        else:
            ts_lines_xx += [f"<{tgt_lang}>-<{src_lang}>|{line}" for line in ts_lines_tgt_tmp]
            ts_lines_yy += ts_lines_src_tmp

    # Shuffle lines in order
    print("Shuffling lines...")
    tr_lines_xx, tr_lines_yy = shuffle_in_order(tr_lines_xx, tr_lines_yy)
    vl_lines_xx, vl_lines_yy = shuffle_in_order(vl_lines_xx, vl_lines_yy)
    ts_lines_xx, ts_lines_yy = shuffle_in_order(ts_lines_xx, ts_lines_yy)

    # Write files
    print("Writing files...")
    write_file_lines(tr_lines_xx, filename=os.path.join(output_path, f"train.{src_alias}"))
    write_file_lines(tr_lines_yy, filename=os.path.join(output_path, f"train.{trg_alias}"))
    write_file_lines(vl_lines_xx, filename=os.path.join(output_path, f"val.{src_alias}"))
    write_file_lines(vl_lines_yy, filename=os.path.join(output_path, f"val.{trg_alias}"))
    write_file_lines(ts_lines_xx, filename=os.path.join(output_path, f"test.{src_alias}"))
    write_file_lines(ts_lines_yy, filename=os.path.join(output_path, f"test.{trg_alias}"))

if __name__ == "__main__":
    main()
    print("Done!")

