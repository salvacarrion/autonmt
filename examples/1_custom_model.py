from autonmt.preprocessing import DatasetBuilder
from autonmt.bundle.report import generate_report

from autonmt.toolkits import AutonmtTranslator
from autonmt.modules.models import Transformer

import os
import datetime
from autonmt.bundle.utils import create_logger


# Set output path/logger
output_path = f".outputs/autonmt/{str(datetime.datetime.now())}"
logger = create_logger(os.path.join(output_path, "logs"))

logger.info("ascustomds")

def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/salva/datasets/",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("10k", 10000)]},
        ],
        subword_models=["word"],
        vocab_sizes=[1000],
        merge_vocabs=False,
        force_overwrite=False
    ).build(make_plots=False, safe=True)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_ds()
    ts_datasets = builder.get_ds(ignore_variants=True)

    # Train & Score a model for each dataset
    scores = []
    for ds in tr_datasets:
        model = AutonmtTranslator(model=Transformer, model_ds=ds, force_overwrite=True)
        model.fit(max_epochs=75, batch_size=128, seed=1234, num_workers=16, patience=10)
        m_scores = model.predict(ts_datasets, metrics={"bleu", "chrf", "ter"}, beams=[1])
        scores.append(m_scores)

    # Make report and print it
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="beam1__sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
