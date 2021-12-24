import autonmt as al

from autonmt import DatasetBuilder
from autonmt.tasks.translation.models import Transformer
from autonmt.tasks.translation.bundle.report import generate_report


def main():
    # Create datasets for training
    tr_datasets = DatasetBuilder(
        base_path="/home/salva/datasets",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
        ],
        subword_models=["word"],
        vocab_sizes=[8000],
        force_overwrite=False,
        interactive=True,
        use_cmd=False,
        conda_env_name=None,
    ).build(make_plots=False, safe=True)

    # Create datasets for testing
    ts_datasets = tr_datasets

    # Train & Score a model for each dataset
    scores = []
    for ds in tr_datasets:
        model = al.Translator(model=Transformer,
                              model_ds=ds, safe_seconds=2,
                              force_overwrite=False, interactive=True,
                              use_cmd=False,
                              conda_env_name="mltests")  # Conda envs will soon be deprecated
        model.fit(max_epochs=5)
        eval_scores = model.predict(ts_datasets, metrics={"bleu", "chrf", "ter"}, beams=[1])
        scores.append(eval_scores)

    # Make report
    generate_report(scores=scores, metric_id="beam_1__sacrebleu_bleu", output_path=".outputs/autonmt",
                    save_figures=True, show_figures=False)


if __name__ == "__main__":
    main()
