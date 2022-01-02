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
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            # {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
        ],
        subword_models=["word"],
        vocab_sizes=[4000],
        merge_vocabs=False,
        force_overwrite=False,
        use_cmd=True,
        eval_mode="same",
        conda_env_name="mltests",
        letter_case="lower",
    ).build(make_plots=False, safe=True)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_ds()
    ts_datasets = builder.get_ds(ignore_variants=True)

    # Train & Score a model for each dataset
    scores = []
    errors = []
    for ds in tr_datasets:
        # try:
        # Instantiate vocabs and model
        max_length = 100
        src_vocab = Vocabulary(max_tokens=max_length).build_from_ds(ds=ds, lang=ds.src_lang)
        trg_vocab = Vocabulary(max_tokens=max_length).build_from_ds(ds=ds, lang=ds.trg_lang)
        model = Transformer(encoder_embed_dim=256, decoder_embed_dim=256,
                            src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

        # Train model
        run_prefix = "model256emb"
        model = AutonmtTranslator(run_prefix=run_prefix, model=model, src_vocab=src_vocab, trg_vocab=trg_vocab, force_overwrite=False)
        model.fit(train_ds=ds, max_epochs=5, batch_size=128, seed=1234, num_workers=16, patience=10)
        m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[1], max_gen_length=max_length)
        scores.append(m_scores)
        print(m_scores)
        # except Exception as e:
        #     print(ds)
        #     print(e)
        #     errors.append((str(ds), str(e)))

    # Make report and print it
    output_path = f".outputs/autonmt/{str(datetime.datetime.now())}"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="beam1__sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))

    print(f"Errors: {len(errors)}")
    print(errors)


if __name__ == "__main__":
    main()
