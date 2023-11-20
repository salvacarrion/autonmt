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
normalize_fn = lambda x: normalize_lines(x, seq=[NFKC(), Strip(), Lowercase()])
preprocess_raw_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn, min_len=1, max_len=None, remove_duplicates=False, shuffle_lines=False)
preprocess_splits_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn, shuffle_lines=True)
preprocess_predict_fn = lambda x: preprocess_lines(x, normalize_fn=normalize_fn)

# BASE_PATH = "/Users/salvacarrion/Documents/Programming/datasets/translate"  # Local
BASE_PATH2 = "/home/scarrion/datasets/translate"  # Remote
BASE_PATH3 = "/app/data"  # Docker
BASE_PATH = BASE_PATH2 if os.environ.get("DEBUG", 0) else BASE_PATH3

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
            {"name": "scielo/health", "languages": ["en-es"], "sizes": [("100k", 100000)]},
            {"name": "scielo/biological", "languages": ["en-es"], "sizes": [("100k", 100000)]},
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

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # return
    # builder_ts = DatasetBuilder(
    #     # Root folder for datasets
    #     base_path=BASE_PATH,
    #
    #     # Set of datasets, languages, training sizes to try
    #     datasets=[
    #         {"name": "multi30k/neutral", "languages": ["en-es"], "sizes": [("original", None)]},
    #         {"name": "multi30k/informal", "languages": ["en-es"], "sizes": [("original", None)]},
    #         {"name": "multi30k/formal", "languages": ["en-es"], "sizes": [("original", None)]},
    #     ],
    # )
    # # Create preprocessing for training and testing
    # tr_datasets = builder.get_train_ds()
    # ts_datasets = builder_ts.get_test_ds()

    # Train & Score a model for each dataset
    scores = []
    for i, train_ds in enumerate(tr_datasets, 1):
        # Instantiate vocabs and model
        src_vocab = Vocabulary(max_tokens=250).build_from_ds(ds=train_ds, lang=train_ds.src_lang)
        trg_vocab = Vocabulary(max_tokens=250).build_from_ds(ds=train_ds, lang=train_ds.trg_lang)
        model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

        # # Load checkpoint
        # path = os.path.join(BASE_PATH, "multi30k/neutral/en-es/original/models/autonmt/runs/multi30k-neutral_en-es_bpe+bytes_8000/checkpoints")
        # checkpoint_path = os.path.join(path, "epoch=013-val_loss=1.429__best.pt")  # Balanced
        # if checkpoint_path:
        #     print(f"\t- Loading previous checkpoint: {checkpoint_path}")
        #     model_state_dict = torch.load(checkpoint_path)
        #     model_state_dict = model_state_dict.get("state_dict", model_state_dict)
        #     model.load_state_dict(model_state_dict)

        # Define trainer
        runs_dir = train_ds.get_runs_path(toolkit="autonmt")
        run_prefix = '_'.join(train_ds.id()[:2]).replace('/', '-')
        run_name = train_ds.get_run_name(run_prefix=run_prefix)  #+ f"__{int(time.time())}"
        trainer = AutonmtTranslator(model=model, src_vocab=src_vocab, trg_vocab=trg_vocab,
                                    runs_dir=runs_dir, run_name=run_name)

        # Print info
        print(f"=> Training model...")
        print(f"\t- TRAINING ({i}/{len(tr_datasets)}): {str(train_ds)}")
        print(f"\t- TESTING ({len(ts_datasets)}): {', '.join([str(x) for x in ts_datasets])}")
        print(f"\t- MODEL PREFIX: {run_prefix}")

        # Train model
        wandb_params = dict(project="continual-learning-new", entity="salvacarrion", reinit=True)
        trainer.fit(train_ds, max_epochs=100, learning_rate=0.001, optimizer="adam", batch_size=256, seed=1234,
                    patience=10, num_workers=0, accelerator="auto", strategy="auto", save_best=True, save_last=True, print_samples=1,
                    wandb_params=wandb_params)

        # Test model
        m_scores = trainer.predict(ts_datasets, metrics={"bleu", "chrf", "ter"}, beams=[1], load_checkpoint="best",
                                   preprocess_fn=preprocess_predict_fn, eval_mode="compatible", force_overwrite=False)
        for ms in m_scores:
            ms['train_dataset'] = str(train_ds)
        scores.append(m_scores)

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
