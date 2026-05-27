"""Five-minute quickstart: HuggingFace dataset → trained model → BLEU score.

Run with:  python examples/0_quickstart_hf.py

Downloads multi30k (de→en, ~29K sentences) from HuggingFace, trains a small
Transformer for 3 epochs, and prints the BLEU score. Designed to fit on a
single laptop GPU or modest CPU.

Optional deps:  pip install datasets
"""
import datetime
import os

from tokenizers.normalizers import NFKC, Strip

from autonmt.reporting.report import format_summary_table, generate_report
from autonmt.core.models import Transformer
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.datasets.processors import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.backends import AutonmtTranslator
from autonmt.backends.base.config import FitConfig, PredictConfig
from autonmt.vocabularies import Vocabulary

BASE_PATH = "datasets/quickstart"
DATASET = "multi30k"
LANG_PAIR = "de-en"


def normalize(x):
    return normalize_lines(x, seq=[NFKC(), Strip()])


def preprocess_train(data, ds):
    return preprocess_pairs(data["src"]["lines"], data["trg"]["lines"], normalize_fn=normalize)


def preprocess_predict(data, ds):
    return preprocess_lines(data["lines"], normalize_fn=normalize)


def main():
    # 1. Fetch & materialise the dataset on disk in AutoNMT layout
    download_hf_dataset(
        hf_id="bentrevett/multi30k", base_path=BASE_PATH,
        dataset_name=DATASET, lang_pair=LANG_PAIR,
        src_field="de", trg_field="en",
    )

    # 2. Run preprocessing + train SentencePiece BPE-4000
    builder = DatasetBuilder(
        base_path=BASE_PATH,
        datasets=[{
            "name": DATASET,
            "languages": [LANG_PAIR],
            "sizes": [("original", None)],
        }],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [4000]}],
        preprocess_raw_fn=preprocess_train,
        preprocess_splits_fn=preprocess_train,
        merge_vocabs=False,
    ).build(force_overwrite=False)

    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # 3. Train one Transformer per dataset variant (one here) and score it
    scores = []
    for train_ds in tr_datasets:
        src_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.src_lang)
        trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.trg_lang)
        model = Transformer(
            src_vocab_size=len(src_vocab),
            trg_vocab_size=len(trg_vocab),
            padding_idx=src_vocab.pad_id,
        )

        # runs_dir = where all runs for this dataset live   (.../<size>/models/autonmt/runs/)
        # run_name = subfolder for THIS run                 (e.g. "quickstart_bpe_4000")
        # Together they give every grid cell its own checkpoints/logs/eval tree.
        trainer = AutonmtTranslator(
            model=model, src_vocab=src_vocab, trg_vocab=trg_vocab,
            runs_dir=train_ds.get_runs_path(toolkit="autonmt"),
            run_name=train_ds.get_run_name(run_prefix="quickstart"),
        )

        # trainer.fit(
        #     train_ds,
        #     config=FitConfig(max_epochs=3, batch_size=128, learning_rate=1e-3, seed=42),
        # )

        scores.append(trainer.predict(
            ts_datasets,
            config=PredictConfig(
                metrics={"bleu"}, beams=[5],
                load_checkpoint="best",
                preprocess_fn=preprocess_predict,
                eval_mode="compatible",
            ),
        ))

    # 4. Save JSON/CSV report + summary table
    output_path = f".outputs/quickstart/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    _, df_summary = generate_report(scores=scores, output_path=output_path)
    print(f"\nReport saved to: {os.path.abspath(output_path)}\n")
    print(format_summary_table(df_summary))


if __name__ == "__main__":
    main()
