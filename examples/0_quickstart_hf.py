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

from autonmt.bundle.report import generate_report
from autonmt.modules.models import Transformer
from autonmt.preprocessing import DatasetBuilder
from autonmt.preprocessing.hf_loader import download_hf_dataset
from autonmt.preprocessing.processors import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.toolkits import AutonmtTranslator
from autonmt.toolkits.config import FitConfig, PredictConfig
from autonmt.vocabularies import Vocabulary

BASE_PATH = "datasets/quickstart"
DATASET = "multi30k"
LANG_PAIR = "de-en"


def normalize(x):
    return normalize_lines(x, seq=[NFKC(), Strip()])


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
        preprocess_raw_fn=lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize),
        preprocess_splits_fn=lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize),
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

        trainer = AutonmtTranslator(
            model=model, src_vocab=src_vocab, trg_vocab=trg_vocab,
            runs_dir=train_ds.get_runs_path(toolkit="autonmt"),
            run_name=train_ds.get_run_name(run_prefix="quickstart"),
        )

        trainer.fit(
            train_ds,
            config=FitConfig(max_epochs=3, batch_size=128, learning_rate=1e-3, seed=42),
        )

        scores.append(trainer.predict(
            ts_datasets,
            config=PredictConfig(
                metrics={"bleu"}, beams=[1],
                load_checkpoint="best",
                preprocess_fn=lambda x: preprocess_lines(x, normalize_fn=normalize),
                eval_mode="compatible",
            ),
        ))

    # 4. Save JSON/CSV report + summary table
    output_path = f".outputs/quickstart/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    _, df_summary = generate_report(scores=scores, output_path=output_path)
    print(f"\nReport saved to: {os.path.abspath(output_path)}\n")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
