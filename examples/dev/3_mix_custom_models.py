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


def merge_pytorch_models(ratio_a, model_a, ratio_b, model_b, model_c):
    for name, param in model_c.named_parameters():
        if name.startswith("decoder"):
            param.data = ratio_a * param.data + ratio_b * model_b.state_dict()[name]
    return model_c


def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        # Root folder for datasets
        base_path=BASE_PATH,

        # Set of datasets, languages, training sizes to try
        datasets=[
            {"name": "multi30k/neutral", "languages": ["en-es"], "sizes": [("original", None)], "split_sizes": (None, 1014, 1000)},
            # {"name": "multi30k/informal", "languages": ["en-es"], "sizes": [("original", None)], "split_sizes": (None, 1014, 1000)},
            # {"name": "multi30k/formal", "languages": ["en-es"], "sizes": [("original", None)], "split_sizes": (None, 1014, 1000)},
            # {"name": "multi30k/neutral-formal", "languages": ["en-es"], "sizes": [("original", None)]},
            # {"name": "multi30k/neutral-informal", "languages": ["en-es"], "sizes": [("original", None)]},
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
    ).build(make_plots=True, force_overwrite=False)

    builder_ts = DatasetBuilder(
        # Root folder for datasets
        base_path=BASE_PATH,

        # Set of datasets, languages, training sizes to try
        datasets=[
            {"name": "multi30k/neutral", "languages": ["en-es"], "sizes": [("original", None)], "split_sizes": (None, 1014, 1000)},
            {"name": "multi30k/informal", "languages": ["en-es"], "sizes": [("original", None)], "split_sizes": (None, 1014, 1000)},
            {"name": "multi30k/formal", "languages": ["en-es"], "sizes": [("original", None)], "split_sizes": (None, 1014, 1000)},
        ],
    )

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder_ts.get_test_ds()

    train_ds = tr_datasets[0]
    for ratio in [1.0, 0.75, 0.5, 0.25, 0.0]:
        ratio_a = ratio
        ratio_b = 1.0 - ratio

        # Instantiate vocabs and model
        src_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.src_lang)
        trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.trg_lang)
        model_a = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)
        model_b = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)
        model_c = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

        # Load checkpoint A: Formal
        path_a = os.path.join(BASE_PATH, "multi30k/neutral-formal/en-es/original/models/autonmt/runs/ft__multi30k-neutral-formal_en-es_bpe+bytes_8000/checkpoints")
        checkpoint_path_a = os.path.join(path_a, "epoch=001-val_loss=1.324__best.pt")
        if checkpoint_path_a:
            print(f"\t- Loading previous checkpoint Formal: {checkpoint_path_a}")
            model_state_dict = torch.load(checkpoint_path_a)
            model_state_dict = model_state_dict.get("state_dict", model_state_dict)
            model_a.load_state_dict(model_state_dict)

        # Load checkpoint B: Informal
        path_b = os.path.join(BASE_PATH, "multi30k/neutral-informal/en-es/original/models/autonmt/runs/ft__multi30k-neutral-informal_en-es_bpe+bytes_8000/checkpoints")
        checkpoint_path_b = os.path.join(path_b, "epoch=001-val_loss=1.331__best.pt")
        if checkpoint_path_b:
            print(f"\t- Loading previous checkpoint Formal: {checkpoint_path_b}")
            model_state_dict = torch.load(checkpoint_path_b)
            model_state_dict = model_state_dict.get("state_dict", model_state_dict)
            model_b.load_state_dict(model_state_dict)

        model_mixed = merge_pytorch_models(ratio_a, model_a, ratio_b, model_b, model_c)

        # Define trainer
        runs_dir = train_ds.get_runs_path(toolkit="autonmt")
        run_prefix = '_'.join(train_ds.id()[:2]).replace('/', '-')
        run_name = f"{ratio_a:.2f}xformal+{ratio_b:.2f}informal__" + train_ds.get_run_name(run_prefix=run_prefix)
        trainer = AutonmtTranslator(model=model_mixed, src_vocab=src_vocab, trg_vocab=trg_vocab,
                                    runs_dir=runs_dir, run_name=run_name)

        # Print info
        print(f"=> Training model...")
        print(f"\t- TESTING ({len(ts_datasets)}): {', '.join([str(x) for x in ts_datasets])}")
        print(f"\t- MODEL PREFIX: {run_prefix}")

        # # Train model
        # wandb_params = dict(project="continual-learning", entity="salvacarrion", reinit=True)
        # comet_params = None  #dict(api_key="SPbJIBtSiGmnWI9Pc7ZuDJ4Wc", project_name="continual-learning", workspace="salvacarrion")
        # trainer.fit(train_ds, max_epochs=100, learning_rate=0.001, optimizer="adamw", batch_size=512, seed=1234,
        #             patience=10, num_workers=4, accelerator="auto", strategy="auto", save_best=True, save_last=True, print_samples=1,
        #             wandb_params=wandb_params, comet_params=comet_params)

        # Test model
        m_scores = trainer.predict(ts_datasets, metrics={"bleu"}, beams=[1], load_checkpoint=None,
                                   preprocess_fn=preprocess_predict_fn, eval_mode="compatible", force_overwrite=True)
        scores = [m_scores]

    # Make report
    output_path = os.path.join(BASE_PATH, f".outputs/autonmt/{str(datetime.datetime.now())}")
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