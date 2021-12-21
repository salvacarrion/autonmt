import os
import autonlp as al
from autonlp import DatasetBuilder


def main():
    base_path = "/home/salva/Downloads"

    # Create splits

    # Define builder and create datasets
    datasets = DatasetBuilder(
        base_path=base_path,
        datasets=[
            {"name": "ccaligned", "languages": ["lg-en"], "sizes": [("original", None), ("10k", 10000)]},
        ],
        subword_models=["word", "unigram"],
        vocab_sizes=[8000],
        force_overwrite=False,
        interactive=True,
    ).build(make_plots=True, safe=True)

    # Train models
    for ds in datasets:
        # Define translator
        model = al.Translator(ds, engine="fairseq")

        # Train & Score
        model.preprocess()
        model.train()
        model.evaluate(eval_datasets=datasets, beams=[1, 5])
        model.score(eval_datasets=datasets, metrics={"bleu", "chrf", "ter" "bertscore", "comet", "beer"})

        # Make plots
        model.make_plots()


if __name__ == "__main__":
    main()
