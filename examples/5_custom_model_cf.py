import datetime

from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Strip, Lowercase

from autonmt.bundle.report import generate_report
from autonmt.modules.models import Transformer
from autonmt.preprocessing import DatasetBuilder
from autonmt.toolkits import AutonmtTranslator
from autonmt.vocabularies import Vocabulary


def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            # {"name": "cf", "languages": ["es-en", "fr-en", "de-en"], "sizes": [("100k", 100000)]},
            # {"name": "cf", "languages": ["es-en", "fr-en", "de-en"], "sizes": [("100k", 100000)]},
            # {"name": "cf", "languages": ["es-en"], "sizes": [("1k", 1000)], "split_sizes": (None, 100, 100)},
            {"name": "cf/multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
        ],
        encoding=[
            # {"subword_models": ["word", "unigram+bytes", "char+bytes"], "vocab_sizes": [8000, 16000]},
            # {"subword_models": ["word", "unigram+bytes"], "vocab_sizes": [8000, 16000, 32000]},
            # {"subword_models": ["char", "unigram+bytes"], "vocab_sizes": [8000]},
            {"subword_models": ["word"], "vocab_sizes": [8000]},
        ],
        normalizer=normalizers.Sequence([NFKC(), Strip(), Lowercase()]),
        merge_vocabs=False,
        eval_mode="compatible",
    ).build(make_plots=False, force_overwrite=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # Train & Score a model for each dataset
    scores = []
    for train_ds in tr_datasets:
        # Instantiate vocabs and model
        src_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.src_lang)
        trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.trg_lang)
        model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

        # Train model
        model = AutonmtTranslator(model=model, src_vocab=src_vocab, trg_vocab=trg_vocab)
        model.fit(train_ds, max_epochs=10, learning_rate=0.001, optimizer="adam", batch_size=128, seed=1234, patience=10, num_workers=12, strategy="dp", force_overwrite=False)
        m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[1], load_best_checkpoint=True, force_overwrite=False, model_ds=train_ds)
        scores.append(m_scores)

    # Make report and print it
    output_path = f".outputs/autonmt/{str(datetime.datetime.now())}"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="beam1__sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
