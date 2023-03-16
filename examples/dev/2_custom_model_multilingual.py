import datetime
import os
from autonmt.bundle.utils import make_dir
import torch

from autonmt.bundle.report import generate_report
from autonmt.modules.models import Transformer
from autonmt.preprocessing import DatasetBuilder
from autonmt.toolkits import AutonmtTranslator
from autonmt.vocabularies import Vocabulary

from autonmt.preprocessing.processors import preprocess_pairs, preprocess_lines, normalize_lines
from tokenizers.normalizers import NFKC, Strip, Lowercase

# Preprocess functions
normalize_fn = lambda x: normalize_lines(x, seq=[NFKC(), Strip(), Lowercase()])
preprocess_splits_fn = lambda x, y: preprocess_pairs(x, y, normalize_fn=normalize_fn)
preprocess_predict_fn = lambda x: preprocess_lines(x, normalize_fn=normalize_fn)

def _get_rehearsal_data(x, y, valid_pairs, past_pairs, ratio_past_data):
    _x, _y = [], []
    for l1, l2 in zip(x, y):
        code = l1[1:3]
        if code in valid_pairs:  # Add new data (100%)
            _x.append(l1)
            _y.append(l2)
        else:  # Add past data (ratio)
            if code in past_pairs and torch.rand(1) < ratio_past_data:
                _x.append(l1)
                _y.append(l2)
    return _x, _y

def filter_data_fn(x, y, split_name, valid_pairs, past_pairs, ratio_past_data, from_fn=None, **kwargs):
    if not valid_pairs: # Add all
        return x, y
    else:
        valid_pairs = set(valid_pairs)
        if from_fn in {"translate"}:  # Raw data
            x, y = zip(*[(l1, l2) for l1, l2 in zip(x, y) if l1[0:2] in valid_pairs])
        else: # Encoded data
            if split_name in {"train"}:
                # x, y = _get_rehearsal_data(x, y, valid_pairs, past_pairs, ratio_past_data)
                x, y = zip(*[(l1, l2) for l1, l2 in zip(x, y) if l1[10:12] in valid_pairs])
            else:  # Dev or test (add all)
                x, y = zip(*[(l1, l2) for l1, l2 in zip(x, y) if l1[10:12] in valid_pairs])  # euro [10:12]; scielo [1:3]
        return x, y

def _gen_filter_data_fn(split_name, valid_pairs=None, past_pairs=None, ratio_past_data=None):
    fn_name = 'xx' if valid_pairs is None else '+'.join(valid_pairs)
    return fn_name, (lambda x, y,
                            split_name=split_name,
                            valid_pairs=valid_pairs, past_pairs=past_pairs, ratio_past_data=ratio_past_data,
                            **kwargs: filter_data_fn(x, y, split_name, valid_pairs, past_pairs, ratio_past_data, **kwargs))

def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        # Root folder for datasets
        base_path="/home/scarrion/datasets/translate",

        # Set of datasets, languages, training sizes to try
        datasets=[
            # {"name": "scielo/_merged", "languages": ["en-xx"], "sizes": [("original", None)]}, #("10k", 10000)
            {"name": "europarl", "languages": ["en-xx"], "sizes": [("original", None)]}, #("10k", 10000)
        ],

        # Set of subword models and vocab sizes to try
        encoding=[
            {"subword_models": ["bytes"], "vocab_sizes": [1000]},
        ],

        # Preprocessing functions
        preprocess_raw_fn=None,
        preprocess_splits_fn=preprocess_splits_fn,
        preprocess_predict_fn=preprocess_predict_fn,

        # Additional args
        merge_vocabs=True,
        eval_mode="same",
    ).build(make_plots=True, force_overwrite=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # Instantiate vocabs and model
    default_ds = tr_datasets[0]
    src_vocab = trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=default_ds, lang=default_ds.dataset_lang_pair)
    # checkpoint_path = 'mymodels/single/1_xx_last.pt'

    # Filter pairs
    past_pairs = {}  # Past training data
    tr_pairs = [None, ["es"], ["fr"], ["de"], ["cs"]]  # Training data
    ts_pairs = [None, ["es"], ["fr"], ["de"], ["cs"]]  # For each model

    # Train and test models
    scores = []
    for i, new_pairs in enumerate(tr_pairs):
        for ratio_past_data in [0.0]:  # [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
            tr_pairs_str = 'xx' if new_pairs is None else '+'.join(new_pairs)
            ts_pairs_str = '|'.join(["xx" if x is None else '+'.join(x) for x in ts_pairs])
            prefix = f"single__tr_{tr_pairs_str}"
            monitor = f'val_{tr_pairs_str}_loss/dataloader_idx_{i}'  # Monitor the loss of the current training data

            # Create path to save the model
            m_path = os.path.join("mymodels/europarl", "single")
            make_dir([m_path])

            print(f"=> Training model...")
            print(f"\t- TRAINING ({i+1}/{len(tr_pairs)}): {tr_pairs_str}")
            print(f"\t- TESTING ({len(ts_pairs)}): {ts_pairs_str}")
            print(f"\t- MODEL PREFIX: {prefix}")
            print(f"\t- MODEL PATH: {m_path}")

            # Set model
            t_model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)
            checkpoint_path = None  #'/home/scarrion/datasets/translate/scielo/_merged/en-xx/original/models/autonmt/runs/single__tr-hh_bpe+bytes_16000/checkpoints/checkpoint_best.pt'
            if checkpoint_path:
                print(f"\t- Loading checkpoint: {checkpoint_path}")
                model_state_dict = torch.load(checkpoint_path)
                model_state_dict = model_state_dict.get("state_dict", model_state_dict)
                t_model.load_state_dict(model_state_dict)

            # Train model
            wandb_params = dict(project="autonmt-europarl", entity="salvacarrion")
            model = AutonmtTranslator(model=t_model, src_vocab=src_vocab, trg_vocab=trg_vocab, wandb_params=wandb_params,
                                        run_prefix=prefix, load_best_checkpoint=False, print_samples=3,
                                        filter_tr_data_fn=_gen_filter_data_fn("train", valid_pairs=new_pairs, past_pairs=past_pairs, ratio_past_data=ratio_past_data),
                                        filter_vl_data_fn=[_gen_filter_data_fn("val", valid_pairs=p) for p in ts_pairs],
                                        filter_ts_data_fn=[_gen_filter_data_fn("test", valid_pairs=p) for p in ts_pairs],
                                        )
            model.fit(default_ds, max_epochs=75, learning_rate=0.001, optimizer="adamw", gradient_clip_val=1.0,  monitor=monitor,
                      batch_size=64, seed=1234, patience=10, num_workers=10, devices="auto", accelerator="auto", strategy="ddp")

            # Save model
            checkpoint_path = os.path.join(m_path, prefix + "_last.pt")
            print(f"\t- Saving current model at: {checkpoint_path}")
            torch.save(t_model.state_dict(), checkpoint_path)

            # m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[1],
            #                          load_best_checkpoint=True, max_len_a=0.0, max_len_b=10,
            #                          model_ds=default_ds)  # model_ds=train_ds => if fit() was not used before
            # scores.append(m_scores)

    # # Make report and print it
    # output_path = f".outputs/autonmt/{str(datetime.datetime.now())}"
    # df_report, df_summary = generate_report(scores=scores, output_path=output_path,
    #                                         plot_metric="translations.xx.beam1.sacrebleu_bleu_score")
    # print("Summary:")
    # print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()
    print("Done!")
