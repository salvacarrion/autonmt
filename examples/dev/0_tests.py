import datetime

from autonmt.bundle.report import generate_report
from autonmt.modules.models import Transformer
from autonmt.preprocessing import DatasetBuilder
from autonmt.toolkits import AutonmtTranslator
from autonmt.vocabularies import Vocabulary

from autonmt.preprocessing.processors import preprocess_pairs, preprocess_lines, normalize_lines
from tokenizers.normalizers import NFKC, Strip

# Preprocess functions
normalize_fn = lambda x: normalize_lines(x, seq=[NFKC(), Strip()])
preprocess_raw_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn, min_len=1, max_len=None, remove_duplicates=False, shuffle_lines=True)
preprocess_splits_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn)
preprocess_predict_fn = lambda x: preprocess_lines(x, normalize_fn=normalize_fn)

def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        # Root folder for datasets
        base_path="/home/scarrion/datasets/translate",

        # Set of datasets, languages, training sizes to try
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
        ],

        # Set of subword models and vocab sizes to try
        encoding=[
            {"subword_models": ["word"], "vocab_sizes": [8000]},
        ],

        # Preprocessing functions
        preprocess_raw_fn=preprocess_raw_fn,
        preprocess_splits_fn=preprocess_splits_fn,
        preprocess_predict_fn=preprocess_predict_fn,

        # Additional args
        merge_vocabs=True,
        eval_mode="same",
    ).build(make_plots=False, force_overwrite=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # Train & Score a model for each dataset
    scores = []
    for train_ds in tr_datasets:
        # Instantiate vocabs and model
        src_vocab = trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.dataset_lang_pair)
        model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

        # Train model
        wandb_params = dict(project="autonmt-cf", entity="salvacarrion")
        model = AutonmtTranslator(model=model, src_vocab=src_vocab, trg_vocab=trg_vocab, wandb_params=wandb_params, skip_val_metrics=True)
        model.fit(train_ds, max_epochs=10, learning_rate=0.0005, optimizer="adam", gradient_clip_val=1.0,
                  batch_size=128, seed=1234, patience=10, num_workers=10, devices="auto", accelerator="gpu", strategy="ddp", precision=32)
        m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[1], load_best_checkpoint=True, model_ds=train_ds)  # model_ds=train_ds => if fit() was not used before
        scores.append(m_scores)

    # Make report and print it
    output_path = f".outputs/autonmt/{str(datetime.datetime.now())}"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="translations.beam1.sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
