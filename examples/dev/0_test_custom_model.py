import datetime
import os
import comet_ml
import torch
torch.set_float32_matmul_precision("high")

from autonmt.bundle.report import generate_report
from autonmt.modules.models import Transformer
from autonmt.preprocessing import DatasetBuilder
from autonmt.toolkits import AutonmtTranslator
from autonmt.vocabularies import Vocabulary

from autonmt.preprocessing.processors import preprocess_pairs, preprocess_lines, normalize_lines
from tokenizers.normalizers import NFKC, Strip, Lowercase

# Preprocess functions
normalize_fn = lambda x: normalize_lines(x, seq=[NFKC(), Strip()])
preprocess_raw_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn, min_len=1, max_len=None, remove_duplicates=False, shuffle_lines=False)
preprocess_splits_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn, shuffle_lines=True)
preprocess_predict_fn = lambda x: preprocess_lines(x, normalize_fn=normalize_fn)

# BASE_PATH1 = "/home/salvacarrion/Documents/datasets/translation"  # Local
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
            {"name": "multi30k/neutral", "languages": ["de-en"], "sizes": [("original", None)], "split_sizes": (None, 1014, 1000)},
            # {"name": "multi30k/informal", "languages": ["de-es"], "sizes": [("original", None)], "split_sizes": (None, 1014, 1000)},
            # {"name": "multi30k/formal", "languages": ["de-es"], "sizes": [("original", None)], "split_sizes": (None, 1014, 1000)},
            # {"name": "multi30k/neutral-formal", "languages": ["en-es"], "sizes": [("original", None)]},
            # {"name": "multi30k/neutral-informal", "languages": ["en-es"], "sizes": [("original", None)]},
            # {"name": "multi30k/merged-neutral-formal-informal", "languages": ["en-es", "de-es"], "sizes": [("original", None)]},
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

    builder_ts = DatasetBuilder(
        # Root folder for datasets
        base_path=BASE_PATH,

        # Set of datasets, languages, training sizes to try
        datasets=[
            {"name": "multi30k/neutral", "languages": ["en-es", "de-es"], "sizes": [("original", None)], "split_sizes": (None, 1014, 1000)},
            # {"name": "multi30k/informal", "languages": ["en-es", "de-es"], "sizes": [("original", None)], "split_sizes": (None, 1014, 1000)},
            # {"name": "multi30k/formal", "languages": ["en-es", "de-es"], "sizes": [("original", None)], "split_sizes": (None, 1014, 1000)},
        ],
    )

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder_ts.get_test_ds()

    # Train & Score a model for each dataset
    scores = []
    for i, train_ds in enumerate(tr_datasets, 1):
        # Instantiate vocabs and model
        src_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.src_lang)
        trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.trg_lang)
        model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

        # Load checkpoint
        # path = os.path.join(BASE_PATH, "multi30k/neutral/en-es/original/models/autonmt/runs/multi30k-neutral_en-es_bpe+bytes_8000/checkpoints")
        # checkpoint_path = os.path.join(path, "epoch=014-val_loss=1.397__best.pt")
        # if checkpoint_path:
        #     print(f"\t- Loading previous checkpoint: {checkpoint_path}")
        #     model_state_dict = torch.load(checkpoint_path)
        #     model_state_dict = model_state_dict.get("state_dict", model_state_dict)
        #     model.load_state_dict(model_state_dict)

        # Define trainer
        runs_dir = train_ds.get_runs_path(toolkit="autonmt")
        run_prefix = '_'.join(train_ds.id()[:2]).replace('/', '-')
        run_name = train_ds.get_run_name(run_prefix=run_prefix)
        trainer = AutonmtTranslator(model=model, src_vocab=src_vocab, trg_vocab=trg_vocab,
                                    runs_dir=runs_dir, run_name=run_name)

        # Print info
        print(f"=> Training model...")
        print(f"\t- TRAINING ({i}/{len(tr_datasets)}): {str(train_ds)}")
        print(f"\t- TESTING ({len(ts_datasets)}): {', '.join([str(x) for x in ts_datasets])}")
        print(f"\t- MODEL PREFIX: {run_prefix}")

        # Train model
        wandb_params = dict(project="continual-learning", entity="salvacarrion", reinit=True)
        comet_params = None  #dict(api_key="SPbJIBtSiGmnWI9Pc7ZuDJ4Wc", project_name="continual-learning", workspace="salvacarrion")
        # trainer.fit(train_ds, max_epochs=100, learning_rate=0.001, optimizer="adamw", batch_size=512, seed=1234,
        #             patience=10, num_workers=0, accelerator="auto", strategy="auto", save_best=True, save_last=True, print_samples=1,
        #             wandb_params=wandb_params, comet_params=comet_params)

        # Test model
        m_scores = trainer.predict(ts_datasets, metrics={"bleu"}, beams=[1], load_checkpoint="best",
                                   preprocess_fn=preprocess_predict_fn, eval_mode="compatible", force_overwrite=False)
        for ms in m_scores:
            ms['train_dataset'] = str(train_ds)
        scores.append(m_scores)

    # Make report and print it
    output_path = os.path.join(BASE_PATH, f".outputs/autonmt/{str(datetime.datetime.now())}")
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="translations.beam1.sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()

    # ##### Reference output #######################################
    # Summary:
    # lang_pair  vocab_size subword_model train_dataset eval_dataset  translations.beam1.sacrebleu_bleu_score
    #     de-en        4000          word  no_specified     multi30k                                33.194409
    #     de-en        4000     bpe+bytes  no_specified     multi30k                                34.062475
    ################################################################