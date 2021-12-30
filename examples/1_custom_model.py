import autonmt as al

from autonmt import DatasetBuilder
from autonmt.modules.nn import Transformer
from autonmt.tasks.translation.bundle.report import generate_report
from autonmt import utils


def main():
    # Create datasets for training
    builder = DatasetBuilder(
        base_path="/home/salva/datasets/",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
        ],
        subword_models=["bytes", "char+bytes", "char", "unigram", "word"],
        vocab_sizes=[8000],
        merge_vocabs=True,
        force_overwrite=False,
    ).build(make_plots=False, safe=True)

    # Create datasets for testing
    tr_datasets = builder.get_ds()
    ts_datasets = builder.get_ds(ignore_variants=True)

    # Train & Score a model for each dataset
    scores = []
    for ds in tr_datasets:
        model = al.Translator(model=Transformer, model_ds=ds, force_overwrite=True)
        model.fit(max_epochs=10, learning_rate=0.001, criterion="cross_entropy", optimizer="adam", clip_norm=1.0,
                  update_freq=1, max_tokens=None, batch_size=64, patience=10, seed=1234, num_gpus=1)
        m_scores = model.predict(ts_datasets, metrics={"bleu", "chrf", "ter"}, beams=[1, 5])
        scores.append(m_scores)

    # Make report and print it
    df_report, df_summary = generate_report(scores=scores, output_path=".outputs", plot_metric="beam1__sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
