import os
import pathlib
import shutil

import autonmt as al

from autonmt.modules.models.transfomer import Transformer
from autonmt.toolkits.autonmt import AutonmtTranslator
from autonmt.toolkits.fairseq import FairseqTranslator
from autonmt.preprocessing.builder import DatasetBuilder
from autonmt.bundle.report import generate_report, summarize_scores

import pytest

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


def train_and_score_autonmt(train_datasets, train_params, test_params, test_datasets):
    # Define model
    model = Transformer

    # Train and score model/s
    scores = []
    for ds in train_datasets:
        model = AutonmtTranslator(model=model, model_ds=ds, safe_seconds=0,
                           force_overwrite=True, interactive=False, use_cmd=False)
        model.fit(**train_params)
        eval_scores = model.predict(test_datasets, **test_params)
        scores.append(eval_scores)
    return scores


def train_and_score_fairseq(train_datasets, train_params, test_params, test_datasets):
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
    for ds in train_datasets:
        model = FairseqTranslator(model_ds=ds, safe_seconds=0,
                                  force_overwrite=True, interactive=False, use_cmd=False,
                                  conda_fairseq_env_name=CONDA_FAIRSEQ_ENV_NAME)
        model.fit(**train_params, fairseq_args=fairseq_args)
        eval_scores = model.predict(test_datasets, **test_params)
        scores.append(eval_scores)
    return scores


def test_autonmt_vs_fairseq(datasets_dir):
    # Add the -s flag to see the prints

    # Create preprocessing for training
    builder = DatasetBuilder(
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

    # Create preprocessing for training and testing
    tr_datasets = builder.get_ds()
    ts_datasets = builder.get_ds(ignore_variants=True)

    # Params
    train_params = dict(max_epochs=1, max_tokens=None, batch_size=128, devices=1)
    test_params = dict(metrics={"bleu"}, beams=[1])
    params = dict(train_datasets=tr_datasets, train_params=train_params,
                  test_datasets=ts_datasets, test_params=test_params)

    # Train models
    fairseq_scores = train_and_score_fairseq(**params)
    autonmt_scores = train_and_score_autonmt(**params)

    # Create output path (debugging
    test_dir = pathlib.Path(__file__).parent.resolve()
    df_fairseq_report, _ = generate_report(scores=fairseq_scores,  output_path=os.path.join(test_dir, ".outputs/test/fairseq"))
    df_autonmt_report, _ = generate_report(scores=autonmt_scores,  output_path=os.path.join(test_dir, ".outputs/test/autonmt"))

    # Summarize results
    df_summary = summarize_scores([df_fairseq_report, df_autonmt_report])

    print("Summary:")
    print(df_summary.to_string(index=False))

    assert True  # Analyze the results manually

    #
    # # Create preprocessing for testing
    # ts_datasets = tr_datasets
    #
    # # Train and score
    # autonmt_score = _test_autonmt(preprocessing)
    # fairseq_score = _test_fairseq(preprocessing)
    # assert abs(autonmt_score - fairseq_score) == approx(2)
