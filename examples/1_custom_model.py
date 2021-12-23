import autonmt as al

from autonmt import DatasetBuilder
from autonmt.tasks.translation.models import Transformer
from autonmt.tasks.translation.bundle.metrics import create_report


def main():
    # Create datasets for training
    tr_datasets = DatasetBuilder(
        base_path="/home/salva/datasets",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
        ],
        subword_models=["word"],
        vocab_sizes=[8000],
    ).build(make_plots=True, safe=True)

    # Create datasets for testing
    ts_datasets = tr_datasets

    # Train & Score a model for each dataset
    scores = []
    for ds in tr_datasets:
        model = al.Translator(model=Transformer, model_ds=ds, safe_seconds=2, force_overwrite=True, interactive=True)
        # model.fit()
        eval_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[5])
        scores.append(eval_scores)

    # Make report
    create_report(scores=scores, metric_id="beam_5__sacrebleu_bleu", output_path=".outputs/autonmt", save_figures=True, show_figures=False)


if __name__ == "__main__":
    main()
