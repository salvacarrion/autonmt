import autonmt as al

from autonmt import DatasetBuilder
from autonmt.modules.nn import Transformer
from autonmt.tasks.translation.bundle.vocabulary import VocabularyBytes
from autonmt.tasks.translation.bundle.report import generate_report
from autonmt import utils


def main():
    # Create datasets for training
    builder = DatasetBuilder(
        base_path="/home/salva/datasets/",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
        ],
        subword_models=["word"],
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
        model = al.Translator(model=Transformer, model_ds=ds, force_overwrite=False)
        model.fit(max_epochs=10, batch_size=128, seed=1234, num_workers=16)
        m_scores = model.predict(ts_datasets, metrics={"bleu", "chrf", "ter"}, beams=[1])
        scores.append(m_scores)

    # Make report and print it
    df_report, df_summary = generate_report(scores=scores, output_path=".outputs", plot_metric="beam1__sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
