import datetime
import time
import os
import torch
torch.set_float32_matmul_precision("high")

from autonmt.modules.models import Transformer
from autonmt.preprocessing import DatasetBuilder
from autonmt.toolkits import AutonmtTranslator
from autonmt.vocabularies import Vocabulary

from autonmt.bundle.report import generate_report
from autonmt.bundle.plots import plot_metrics

from autonmt.preprocessing.processors import preprocess_pairs, preprocess_lines, normalize_lines
from tokenizers.normalizers import NFKC, Strip, Lowercase

# Preprocess functions
normalize_fn = lambda x: normalize_lines(x, seq=[NFKC(), Strip()])
preprocess_raw_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn, min_len=1, max_len=None, remove_duplicates=False, shuffle_lines=False)
preprocess_splits_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn)
preprocess_predict_fn = lambda x: preprocess_lines(x, normalize_fn=normalize_fn)

# BASE_PATH1 = "/home/salvacarrion/Documents/datasets/translation"  # Local
BASE_PATH2 = "/home/scarrion/datasets/translate"  # Remote
BASE_PATH3 = "/app/data"  # Docker
BASE_PATH = BASE_PATH2 if os.environ.get("DEBUG", 0) else BASE_PATH3


def merge_pytorch_models(ratio_a, model_a, ratio_b, model_b, ratio_c, model_c):
    for name, param in model_c.named_parameters():
        param.data = ratio_a * model_a.state_dict()[name] + ratio_b * model_b.state_dict()[name]
    return model_c


def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        # Root folder for datasets
        base_path=BASE_PATH,

        # Set of datasets, languages, training sizes to try
        datasets=[
            # Multi30k
            # {"name": "multi30k/neutral", "languages": ["en-es"], "sizes": [("original", None)]},
            # {"name": "multi30k/neutral-informal", "languages": ["en-es"], "sizes": [("original", None)]},
            # {"name": "multi30k/neutral-formal", "languages": ["en-es"], "sizes": [("original", None)]},

            # Scielo
            # {"name": "scielo/health", "languages": ["en-es"], "sizes": [("100k", 100000), ("50k", 50000)]},
            # {"name": "scielo/biological", "languages": ["en-es"], "sizes": [("100k", 100000), ("50k", 50000)]},
            {"name": "scielo/merged100k", "languages": ["en-es"], "sizes": [("100k", 100000)]},  # Dummy
        ],

        # Set of subword models and vocab sizes to try
        encoding=[
            {"subword_models": ["bpe+bytes"], "vocab_sizes": [8000]},
        ],

        # Preprocessing functions
        preprocess_raw_fn=preprocess_raw_fn,
        preprocess_splits_fn=preprocess_splits_fn,

        # Additional args
        merge_vocabs=False,
    ).build(make_plots=False, force_overwrite=False)

    # # Create preprocessing for training and testing
    # tr_datasets = builder.get_train_ds()
    # ts_datasets = builder.get_test_ds()

    builder_ts = DatasetBuilder(
        # Root folder for datasets
        base_path=BASE_PATH,

        # Set of datasets, languages, training sizes to try
        datasets=[
            # Multi30k
            # {"name": "multi30k/neutral", "languages": ["en-es"], "sizes": [("original", None)]},
            # {"name": "multi30k/informal", "languages": ["en-es"], "sizes": [("original", None)]},
            # {"name": "multi30k/formal", "languages": ["en-es"], "sizes": [("original", None)]},

            # Scielo
            {"name": "scielo/health", "languages": ["en-es"], "sizes": [("original", None)]},
            {"name": "scielo/biological", "languages": ["en-es"], "sizes": [("original", None)]},
        ],
    )
    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder_ts.get_test_ds()

    train_ds = tr_datasets[0]
    scores = []
    for ratio in [0.5]:
        ratio_a = ratio
        ratio_b = 1.0 - ratio
        ratio_c = 0.0

        # Instantiate vocabs and model
        src_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.src_lang)
        trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.trg_lang)
        model_a = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)
        model_b = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)
        model_c = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

        # Load checkpoint A
        size = "50k"
        path_a = os.path.join(BASE_PATH, f"scielo/health/en-es/{size}/models/autonmt/runs/scielo-health_en-es_bpe+bytes_8000/checkpoints")
        checkpoint_path_a = os.path.join(path_a, "epoch=017-val_loss=2.469__best.pt")
        if checkpoint_path_a:
            print(f"\t- Loading previous checkpoint A: {checkpoint_path_a}")
            model_state_dict = torch.load(checkpoint_path_a)
            model_state_dict = model_state_dict.get("state_dict", model_state_dict)
            model_a.load_state_dict(model_state_dict)

        # Load checkpoint B
        path_b = os.path.join(BASE_PATH, f"scielo/biological/en-es/{size}/models/autonmt/runs/scielo-biological_en-es_bpe+bytes_8000/checkpoints")
        checkpoint_path_b = os.path.join(path_b, "epoch=024-val_loss=2.412__best.pt")
        if checkpoint_path_b:
            print(f"\t- Loading previous checkpoint B: {checkpoint_path_b}")
            model_state_dict = torch.load(checkpoint_path_b)
            model_state_dict = model_state_dict.get("state_dict", model_state_dict)
            model_b.load_state_dict(model_state_dict)

        # # Load checkpoint C
        # path_c = os.path.join(BASE_PATH, "scielo100k/merged/en-es/50k/models/autonmt/runs/scielo100k-merged_en-es_bpe+bytes_8000/checkpoints")
        # checkpoint_path_c = os.path.join(path_c, "epoch=021-val_loss=2.578__best.pt")
        # if checkpoint_path_b:
        #     print(f"\t- Loading previous checkpoint C: {checkpoint_path_c}")
        #     model_state_dict = torch.load(checkpoint_path_c)
        #     model_state_dict = model_state_dict.get("state_dict", model_state_dict)
        #     model_c.load_state_dict(model_state_dict)

        # Merge models
        model_mixed = merge_pytorch_models(ratio_a, model_a, ratio_b, model_b, ratio_c, model_c)

        # Define trainer
        runs_dir = train_ds.get_runs_path(toolkit="autonmt")
        run_prefix = '_'.join(train_ds.id()[:2]).replace('/', '-')
        run_name = f"{ratio_a:.2f}xH+{ratio_b:.2f}xB__" + train_ds.get_run_name(run_prefix=run_prefix)
        trainer = AutonmtTranslator(model=model_mixed, src_vocab=src_vocab, trg_vocab=trg_vocab,
                                    runs_dir=runs_dir, run_name=run_name)

        # Print info
        print(f"=> Training model...")
        print(f"\t- TESTING ({len(ts_datasets)}): {', '.join([str(x) for x in ts_datasets])}")
        print(f"\t- MODEL PREFIX: {run_prefix}")

        # Test model
        m_scores = trainer.predict(ts_datasets, metrics={"bleu", "chrf", "ter"}, beams=[1], load_checkpoint=None,
                                   preprocess_fn=preprocess_predict_fn, eval_mode="compatible", force_overwrite=True)
        for ms in m_scores:
            ms['train_dataset'] = str(run_name)
        scores.append(m_scores)

    # Make report
    output_path = os.path.join(BASE_PATH, f".outputs/autonmt/Scielo_Base_Mixed")
    df_report, df_summary = generate_report(scores=scores, output_path=output_path)

    # Print summary
    print("Summary:")
    print(df_summary.to_string(index=False))

    # Plot metrics
    plots_path = os.path.join(output_path, "plots")
    plot_metrics(output_path=plots_path, df_report=df_report, plot_metric="translations.beam1.sacrebleu_bleu_score",
                 xlabel="MT Models", ylabel="BLEU Score", title="Model comparison")


if __name__ == "__main__":
    main()

    # ##### Reference output #######################################
    # Summary:
    # lang_pair  vocab_size subword_model train_dataset eval_dataset  translations.beam1.sacrebleu_bleu_score
    #     de-en        4000          word  no_specified     multi30k                                33.194409
    #     de-en        4000     bpe+bytes  no_specified     multi30k                                34.062475
    ################################################################