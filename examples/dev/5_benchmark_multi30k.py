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

def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        # Root folder for datasets
        base_path=BASE_PATH,

        # Set of datasets, languages, training sizes to try
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
        ],

        # Set of subword models and vocab sizes to try
        encoding=[
            {"subword_models": ["word", "bpe+bytes"], "vocab_sizes": [4000]},
        ],

        # Preprocessing functions
        preprocess_raw_fn=preprocess_raw_fn,
        preprocess_splits_fn=preprocess_splits_fn,

        # Additional args
        merge_vocabs=False,
    ).build(make_plots=False, force_overwrite=True)

    builder_ts = DatasetBuilder(
        # Root folder for datasets
        base_path=BASE_PATH,

        # Set of datasets, languages, training sizes to try
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)], "split_sizes": (None, 1014, 1000)},
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
        run_name = train_ds.get_run_name(run_prefix=run_prefix) + f"__{int(time.time())}"
        trainer = AutonmtTranslator(model=model, src_vocab=src_vocab, trg_vocab=trg_vocab,
                                    runs_dir=runs_dir, run_name=run_name)

        # Print info
        print(f"=> Training model...")
        print(f"\t- TRAINING ({i}/{len(tr_datasets)}): {str(train_ds)}")
        print(f"\t- TESTING ({len(ts_datasets)}): {', '.join([str(x) for x in ts_datasets])}")
        print(f"\t- MODEL PREFIX: {run_prefix}")

        # Train model
        wandb_params = None  #dict(project="continual-learning", entity="salvacarrion", reinit=True)
        comet_params = None  #dict(api_key="SPbJIBtSiGmnWI9Pc7ZuDJ4Wc", project_name="continual-learning", workspace="salvacarrion")
        trainer.fit(train_ds, max_epochs=10, learning_rate=0.001, optimizer="adam", batch_size=128, seed=1234,
                    patience=10, num_workers=0, accelerator="auto", strategy="auto", save_best=True, save_last=True, print_samples=1,
                    wandb_params=wandb_params, comet_params=comet_params)

        # Test model
        m_scores = trainer.predict(ts_datasets, metrics={"bleu"}, beams=[1], load_checkpoint="best",
                                   preprocess_fn=preprocess_predict_fn, eval_mode="compatible", force_overwrite=True)
        for ms in m_scores:
            ms['train_dataset'] = str(train_ds)
        scores.append(m_scores)

    # Make report and print it
    output_path = f".outputs/autonmt/{str(datetime.datetime.now())}"
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

    ################################################################
    # v0.5: [NFKC(), Strip(), Lowercase], Overwrite=True!, no shuffle (split), 128batch, adam, 0.001lr, seed=1234¿?. iter=10
    #     de-en       4000          word      multi30k/neutral_de-en_original_word_4000 multi30k-ori                                34.179303
    #     de-en       4000     bpe+bytes multi30k/neutral_de-en_original_bpe+bytes_4000 multi30k-ori                                34.019237

    # v0.5: [NFKC(), Strip(), NO-Lowercase], Overwrite=True!, no shuffle (split), 128batch, adam, 0.001lr, seed=1234¿?. iter=10
    #     de-en       4000          word      multi30k/neutral_de-en_original_word_4000 multi30k-ori                                34.207387
    #     de-en       4000     bpe+bytes multi30k/neutral_de-en_original_bpe+bytes_4000 multi30k-ori                                33.085672

    # v0.5: [NFKC(), Strip(), NO-Lowercase!], Overwrite=True!, no shuffle (split), 1024batch, adamw, 0.001lr, seed=1234
    # lang_pair vocab_size subword_model                                   train_dataset     eval_dataset  translations.beam1.sacrebleu_bleu_score
    #     de-en       4000          word       multi30k/neutral_de-en_original_word_4000 multi30k/neutral                                32.651625
    #     de-en       8000          word       multi30k/neutral_de-en_original_word_8000 multi30k/neutral                                32.140449
    #     de-en      10000          word      multi30k/neutral_de-en_original_word_10000 multi30k/neutral                                30.840177
    #     de-en       4000     bpe+bytes  multi30k/neutral_de-en_original_bpe+bytes_4000 multi30k/neutral                                32.062014
    #     de-en       8000     bpe+bytes  multi30k/neutral_de-en_original_bpe+bytes_8000 multi30k/neutral                                32.079096
    #     de-en      10000     bpe+bytes multi30k/neutral_de-en_original_bpe+bytes_10000 multi30k/neutral                                32.649577
    #     de-en    357/357    char+bytes multi30k/neutral_de-en_original_char+bytes_1000 multi30k/neutral                                32.819399
    ################################################################
