from autonmt.preprocessing import DatasetBuilder
from autonmt.bundle.report import generate_report

from autonmt.toolkits import AutonmtTranslator
from autonmt.modules.models import Transformer
from autonmt.vocabularies import Vocabulary

import os
import datetime


def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            # {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            # {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
            {"name": "scielo/health", "languages": ["es-en"], "sizes": [("100k", 100000)]},
            {"name": "scielo/biological", "languages": ["es-en"], "sizes": [("100k", 100000)]},
        ],
        subword_models=["word", "unigram+bytes"],
        vocab_sizes=[8000, 16000],
        merge_vocabs=False,
        force_overwrite=False,
        eval_mode="compatible",
        letter_case="lower",
    ).build(make_plots=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # Train & Score a model for each dataset
    scores = []
    for ds in tr_datasets:
        # Instantiate vocabs and model
        src_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=ds, lang=ds.src_lang)
        trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=ds, lang=ds.trg_lang)
        model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

        # Train model
        wandb_params = dict(project="autonmt", entity="salvacarrion")
        model = AutonmtTranslator(model=model, src_vocab=src_vocab, trg_vocab=trg_vocab, model_ds=ds, wandb_params=wandb_params, force_overwrite=True)
        model.fit(max_epochs=300, learning_rate=0.001, optimizer="adam", batch_size=256, seed=1234, patience=10, num_workers=12, strategy="dp")
        m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[1], load_best_checkpoint=True)
        scores.append(m_scores)

    # Make report and print it
    output_path = f".outputs/autonmt/{str(datetime.datetime.now())}"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="beam1__sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
