import datetime
import os
from autonmt.bundle.utils import make_dir

from autonmt.bundle.report import generate_report
from autonmt.modules.models import Transformer
from autonmt.preprocessing import DatasetBuilder
# from autonmt.toolkits import AutonmtTranslator
from autonmt.toolkits import AutonmtTranslator
from autonmt.vocabularies import Vocabulary

from autonmt.preprocessing.processors import preprocess_pairs, preprocess_lines, normalize_lines
from tokenizers.normalizers import NFKC, Strip, Lowercase

# Preprocess functions
normalize_fn = lambda x: normalize_lines(x, seq=[NFKC(), Strip(), Lowercase()])
preprocess_splits_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn)
preprocess_predict_fn = lambda x: preprocess_lines(x, normalize_fn=normalize_fn)

def filter_data_fn(x, y, keys, **kwargs):
    if not keys:
        return x, y
    else:
        keys = set(keys)
        if kwargs.get("from_translate"):
            x, y = zip(*[(l1, l2) for l1, l2 in zip(x, y) if l1[0:2] in keys])  # de-en
        else:
            x, y = zip(*[(l1, l2) for l1, l2 in zip(x, y) if l1[1:3] in keys])  # _de-en
        return x, y

def _gen_filter_data_fn(keys):
    fn_name = 'xx' if keys is None else '+'.join(keys)
    return fn_name, (lambda x, y, keys=keys, **kwargs: filter_data_fn(x, y, keys, **kwargs))

def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        # Root folder for datasets
        base_path="datasets/translate",

        # Set of datasets, languages, training sizes to try
        datasets=[
            {"name": "europarl", "languages": ["en-xx"], "sizes": [("100k", 100000)]}, #("original", None),
        ],

        # Set of subword models and vocab sizes to try
        encoding=[
            {"subword_models": ["bpe+bytes"], "vocab_sizes": [16000]},
        ],

        # Preprocessing functions
        preprocess_raw_fn=None,
        preprocess_splits_fn=preprocess_splits_fn,
        preprocess_predict_fn=preprocess_predict_fn,

        # Additional args
        merge_vocabs=True,
        eval_mode="same",
    ).build(make_plots=False, force_overwrite=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # Instantiate vocabs and model
    default_ds = tr_datasets[0]
    src_vocab = trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=default_ds, lang=default_ds.dataset_lang_pair)
    checkpoint_path = None #'mymodels/single/1_xx_last.pt'

    # Filter pairs
    filter_tr_pairs = [None, ["en-es"], ["en-fr", "en-de"], ["en-cs"]]  # Training data
    filter_ts_pairs = [None, ["en-es"], ["en-fr"], ["en-de"], ["en-cs"]]  # For each model

    # Train and test models
    scores = []
    for i, filter_tr_keys_i in enumerate(filter_tr_pairs):
        tr_pairs_str = 'xx' if filter_tr_keys_i is None else '+'.join(filter_tr_keys_i)
        ts_pairs_str = '|'.join(["xx" if x is None else '+'.join(x) for x in filter_ts_pairs])
        prefix = f"{i+1}_modelTrainedOn-{tr_pairs_str}"
        monitor = f'val_{tr_pairs_str}_loss/dataloader_idx_{i}'  # Monitor the loss of the current training data

        # Create path to save the model
        m_path = os.path.join("mymodels", "single")
        make_dir([m_path])

        print(f"=> Training model...")
        print(f"\t- TRAINING ({i+1}/{len(filter_tr_pairs)}): {tr_pairs_str}")
        print(f"\t- TESTING ({len(filter_ts_pairs)}): {ts_pairs_str}")
        print(f"\t- MODEL PREFIX: {prefix}")
        print(f"\t- MODEL PATH: {m_path}")

        # Set model
        t_model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)
        # if checkpoint_path:
        #     print(f"\t- Loading checkpoint: {checkpoint_path}")
        #     model_state_dict = torch.load(checkpoint_path)
        #     model_state_dict = model_state_dict.get("state_dict", model_state_dict)
        #     t_model.load_state_dict(model_state_dict)

        # Train model
        wandb_params = None  # dict(project="autonmt", entity="salvacarrion")
        model = AutonmtTranslator(model=t_model, src_vocab=src_vocab, trg_vocab=trg_vocab, wandb_params=wandb_params,
                                    run_prefix=prefix, load_best_checkpoint=False, print_samples=3,
                                    filter_tr_data_fn=_gen_filter_data_fn(filter_tr_keys_i),
                                    filter_vl_data_fn=[_gen_filter_data_fn(keys) for keys in filter_ts_pairs],
                                    filter_ts_data_fn=[_gen_filter_data_fn(keys) for keys in filter_ts_pairs],
                                    )
        model.fit(default_ds, max_epochs=1, learning_rate=0.0001, optimizer="adamw", batch_size=64, seed=1234,
                  patience=10, num_workers=1, monitor=monitor, devices="auto", accelerator="auto", strategy="ddp")

        # # Save model
        # checkpoint_path = os.path.join(m_path, prefix + "_last.pt")
        # print(f"\t- Saving current model at: {checkpoint_path}")
        # torch.save(t_model.state_dict(), checkpoint_path)

        asd = 3
        m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[1],
                                 load_best_checkpoint=True, max_len_a=0.0, max_len_b=10,
                                 model_ds=default_ds)  # model_ds=train_ds => if fit() was not used before
        scores.append(m_scores)

    # Make report and print it
    output_path = f".outputs/autonmt/{str(datetime.datetime.now())}"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path,
                                            plot_metric="translations.xx.beam1.sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()
