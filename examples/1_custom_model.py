from autonmt.preprocessing import DatasetBuilder
from autonmt.bundle.report import generate_report

from autonmt.toolkits import AutonmtTranslator
from autonmt.modules.models import Transformer

import os
import datetime


def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/salva/datasets/",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            # {"name": "iwlst16", "languages": ["de-en"], "sizes": [("100k", 100000)]},
            # {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
        ],
        subword_models=["word"],
        vocab_sizes=[1000, 2000, 4000, 8000],
        merge_vocabs=False,
        force_overwrite=False,
        use_cmd=True,
        eval_mode="same",
    ).build(make_plots=True, safe=True)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_ds()
    ts_datasets = builder.get_ds(ignore_variants=True)

    # Train & Score a model for each dataset
    scores = []
    errors = []
    for ds in tr_datasets:
        try:
            model = AutonmtTranslator(model=Transformer, model_ds=ds, force_overwrite=False)
            model.fit(max_epochs=75, batch_size=128, seed=1234, num_workers=16, patience=10)
            m_scores = model.predict(ts_datasets, metrics={"bleu", "chrf", "ter"}, beams=[1])
            scores.append(m_scores)
        except Exception as e:
            print(ds)
            print(e)
            errors.append((str(ds), str(e)))

    # Make report and print it
    output_path = f".outputs/autonmt/{str(datetime.datetime.now())}"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="beam1__sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))

    print(f"Errors: {len(errors)}")
    print(errors)


if __name__ == "__main__":
    main()
