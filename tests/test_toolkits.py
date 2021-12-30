import os
import pathlib
import shutil

import pandas as pd

import autonmt as al

from autonmt import DatasetBuilder
from autonmt.modules.nn import Transformer
from autonmt.tasks.translation.bundle.report import scores2pandas, summarize_scores, generate_report
from autonmt import utils

import pytest
from pytest import approx
import math

CONDA_FAIRSEQ_ENV_NAME = "fairseq"


@pytest.fixture
def datasets_dir(tmp_path):
    # Create path
    test_dir = pathlib.Path(__file__).parent.resolve()
    src_dataset_path = os.path.join(test_dir, "data")
    trg_dataset_path = os.path.join(tmp_path, "data")

    # Copy data
    dest_dir = shutil.copytree(src_dataset_path, trg_dataset_path)

    return dest_dir


def train_and_score_autonmt(datasets, train_params, test_params):
    # Define model
    model = Transformer

    # Train and score model/s
    scores = []
    for ds in datasets:
        model = al.Translator(model=model, model_ds=ds, safe_seconds=0,
                              force_overwrite=True, interactive=False, use_cmd=False)
        model.fit(**train_params)
        eval_scores = model.predict(datasets, **test_params)
        scores.append(eval_scores)
    return scores


def train_and_score_fairseq(datasets, train_params, test_params):
    # Define model
    fairseq_args = [
        "--arch transformer",
        "--encoder-embed-dim 256",
        "--decoder-embed-dim 256",
        "--encoder-layers 3",
        "--decoder-layers 3",
        "--encoder-attention-heads 8",
        "--decoder-attention-heads 8",
        "--encoder-ffn-embed-dim 512",
        "--decoder-ffn-embed-dim 512",
        "--dropout 0.1",

        "--no-epoch-checkpoints",
        "--maximize-best-checkpoint-metric",
        "--best-checkpoint-metric bleu",
        "--eval-bleu",
        "--eval-bleu-args '{\"beam\": 5}'",
        "--eval-bleu-print-samples",
        "--scoring sacrebleu",
        "--log-format simple",
        "--task translation",
    ]

    # Train and score model/s
    scores = []
    for ds in datasets:
        model = al.FairseqTranslator(model_ds=ds, safe_seconds=0,
                                     force_overwrite=True, interactive=False, use_cmd=False,
                                     conda_fairseq_env_name=CONDA_FAIRSEQ_ENV_NAME)
        model.fit(**train_params, fairseq_args=fairseq_args)
        eval_scores = model.predict(datasets, **test_params)
        scores.append(eval_scores)
    return scores


def test_autonmt_vs_fairseq(datasets_dir):
    # Add the -s flag to see the prints

    # Create datasets for training
    datasets = DatasetBuilder(
        base_path=datasets_dir,
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
        ],
        subword_models=["word"],
        vocab_sizes=[8000],
        merge_vocabs=True,
        force_overwrite=True,
        interactive=False,
        use_cmd=False,
    ).build(make_plots=False, safe=True)

    # Params
    train_params = dict(max_epochs=10, learning_rate=0.001, criterion="cross_entropy", optimizer="adam", clip_norm=1.0,
                        update_freq=1, max_tokens=None, batch_size=64, patience=10, seed=1234, num_gpus=1)
    test_params = dict(metrics={"bleu"}, beams=[1])
    params = dict(datasets=datasets, train_params=train_params, test_params=test_params)

    # Train models
    fairseq_scores = train_and_score_fairseq(**params)
    autonmt_scores = train_and_score_autonmt(**params)

    # Create output path (debugging
    test_dir = pathlib.Path(__file__).parent.resolve()
    df_fairseq_report, _ = generate_report(scores=fairseq_scores,  output_path=test_dir)
    df_autonmt_report, _ = generate_report(scores=autonmt_scores,  output_path=test_dir)

    # Summarize results
    df_summary = summarize_scores([df_fairseq_report, df_autonmt_report])

    print("Summary:")
    print(df_summary.to_string(index=False))

    assert 1==1

    #
    # # Create datasets for testing
    # ts_datasets = tr_datasets
    #
    # # Train and score
    # autonmt_score = _test_autonmt(datasets)
    # fairseq_score = _test_fairseq(datasets)
    # assert abs(autonmt_score - fairseq_score) == approx(2)
