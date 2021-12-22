import os
import autonlp as al
from autonlp import DatasetBuilder
from autonlp.tasks.translation.base import BaseTranslator
from autonlp.tasks.translation.toolkits.constants import FAIRSEQ_1
from autonlp.tasks.translation.metrics import create_report


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
    scores = {}
    for train_ds in tr_datasets:
        model = al.FairseqTranslator(conda_fairseq_env_name="fairseq", conda_env_name="mltests", force_overwrite=False)
        model.fit(train_ds, fairseq_args=FAIRSEQ_1)
        m_scores = model.predict(ts_datasets, metrics=BaseTranslator.METRICS_SUPPORTED, beams=[5])
        scores[str(train_ds)] = m_scores

    # Make report
    create_report(metrics=scores, metric_id="beam_5__sacrebleu_bleu", output_path=".outputs", save_figures=True, show_figures=False)
    asdsa = 3


if __name__ == "__main__":
    main()
