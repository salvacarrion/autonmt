import datetime
import os

import tqdm

from autonmt.bundle.utils import make_dir
import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import glob

from autonmt.bundle.report import generate_report
from autonmt.modules.models import Transformer
from autonmt.preprocessing import DatasetBuilder
from autonmt.toolkits import AutonmtTranslator
from autonmt.vocabularies import Vocabulary
from autonmt.modules.models.transfomer_grads import TransformerGrads
from autonmt.toolkits.autonmt_grads import AutonmtTranslatorGrads

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
            x, y = zip(*[(l1, l2) for l1, l2 in zip(x, y) if l1[6:8] in valid_pairs])
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

def compute_grads(model, train_tds, num_workers=10, batch_size=64, max_tokens=None):
    print(f"=> Computing gradients...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Prepare criterion and optimizer
    model.configure_criterion("cross_entropy")
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Prepare data
    train_loader = DataLoader(train_tds,
                              collate_fn=lambda x: train_tds.collate_fn(x, max_tokens=max_tokens),
                              num_workers=num_workers, pin_memory=True,
                              batch_size=batch_size, shuffle=False,
                              )

    # Compute grads
    model.train()
    optimizer.zero_grad()

    # accumulating gradients
    for x, y in tqdm.tqdm(train_loader, total=len(train_loader)):
        batch = x.to(device), y.to(device)
        loss, _ = model._step(batch, 0, log_prefix=None)
        loss.backward()

    # Get weights and grads
    weights_dict = {}
    grad_dict = {}
    for name, param in model.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy()
        grad_dict[name] = param.grad.detach().cpu().numpy()

    return weights_dict, grad_dict

def regularization_fn(model, loss, **kwargs):
    d_tasks = kwargs.get("d_tasks")
    reg_type = kwargs.get("reg_type")
    lmda = kwargs.get("lmda")

    # Apply regularization
    if reg_type is None:
        pass
    elif reg_type == "l1":
        for task_id in d_tasks.keys():
            for name, param in model.named_parameters():
                optpar = torch.tensor(d_tasks[task_id]["weights"][name]).to(param.device)
                loss += ((optpar - param).abs()).sum() * lmda
    elif reg_type == "l2":
        for task_id in d_tasks.keys():
            for name, param in model.named_parameters():
                optpar = torch.tensor(d_tasks[task_id]["weights"][name]).to(param.device)
                loss += ((optpar - param).pow(2)).sum() * lmda
    elif reg_type == "ewc":
        for task_id in d_tasks.keys():
            for name, param in model.named_parameters():
                grads = torch.tensor(d_tasks[task_id]["gradients"][name]).to(param.device)
                optpar = torch.tensor(d_tasks[task_id]["weights"][name]).to(param.device)
                fisher = grads.pow(2)
                loss += (fisher * (optpar - param).pow(2)).sum() * lmda
    else:
        raise ValueError(f"Unknown value '{reg_type}' for reg_type")

def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        # Root folder for datasets
        base_path="/home/scarrion/datasets/translate",

        # Set of datasets, languages, training sizes to try
        datasets=[
            # {"name": "scielo/_merged", "languages": ["en-xx"], "sizes": [("original", None)]}, #("10k", 10000)
            # {"name": "europarl", "languages": ["en-xx"], "sizes": [("original", None)]}, #("10k", 10000)
            {"name": "europarl", "languages": ["en-xx"], "sizes": [("40k", 40000)]}, #("10k", 10000)
        ],

        # Set of subword models and vocab sizes to try
        encoding=[
            {"subword_models": ["bpe+bytes"], "vocab_sizes": [16000]},
        ],

        # Preprocessing functions
        preprocess_raw_fn=None,
        preprocess_splits_fn=preprocess_splits_fn,

        # Additional args
        merge_vocabs=True,
    ).build(make_plots=False, force_overwrite=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # Instantiate vocabs and model
    default_ds = tr_datasets[0]
    src_vocab = trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=default_ds, lang=default_ds.dataset_lang_pair)
    # checkpoint_path = 'mymodels/single/1_xx_last.pt'

    # Filter pairs
    tr_pairs = [["es"], ["fr"], ["de"], ["cs"]]  # Training data
    ts_pairs = [None, ["es"], ["fr"], ["de"], ["cs"]]  # For each model
    _ts_pairs = ["xx" if x is None else '+'.join(x) for x in ts_pairs]

    # Train and test models
    counter = 0
    scores = []

    for ratio_past_data in [0.0]: #[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        for reg_type in [None]:

            # Load base model/grads
            d_tasks = {}
            past_pairs = []
            checkpoint_path = None
            sequential_tr = True

            # Load previous checkpoint
            if sequential_tr:
                past_pairs = [["es"]]  # Past training data
                run_prefix = f"cf__tr_[]->[es]+[]x[0.0]__reg_(none)"  # First model
                run_name = default_ds.get_run_name(run_prefix)
                path = default_ds.get_model_checkpoints_path("autonmt", run_name)
                checkpoint_path = os.path.join(path, f'checkpoint_best__epoch=33-val_es_loss-dataloader_idx_1=0.00.pt')
                weights = torch.load(os.path.join(path, f'checkpoint_best__epoch=33-val_es_loss-dataloader_idx_1=0.00.pt'))
                weights1 = torch.load(os.path.join(path, f'checkpoint_last__weights.pt'))
                grads = torch.load(os.path.join(path, f'checkpoint_last__gradients.pt'))
                d_tasks[run_prefix] = {"weights": weights, "gradients": grads}

            for i, new_tr_pairs in enumerate(tr_pairs):
                counter += 1
                wandb_params = dict(project="autonmt-europarl", entity="salvacarrion")

                # Set names
                _past_pairs = [p[0] if p else "xx" for p in past_pairs]
                tr_pairs_old_str = '+'.join(_past_pairs) if past_pairs else ""
                tr_pairs_new_str = '+'.join(new_tr_pairs) if new_tr_pairs else "xx"
                ts_pairs_str = '|'.join(_ts_pairs)

                # Run name
                alias = "xxxcf"
                tr_pairs_str = f"tr_[{tr_pairs_old_str}]->[{tr_pairs_new_str}]"
                tr_pairs_str += f"+[{tr_pairs_old_str}]x[{ratio_past_data}]"# if ratio_past_data else ""
                tr_pairs_str += f"__reg_" + (reg_type.lower().strip() if reg_type else "(none)")
                run_prefix = f"{alias}__{tr_pairs_str}"

                # Set data and stuff
                new_tr_pair_str = new_tr_pairs[0] if new_tr_pairs else "xx" # Trick: Only one pair
                monitor = f'val_{new_tr_pair_str}_loss/dataloader_idx_{_ts_pairs.index(new_tr_pair_str)}'

                # Print info
                print(f"=> Training model...")
                print(f"\t- TRAINING ({i+1}/{len(tr_pairs)}): {tr_pairs_str}")
                print(f"\t- TESTING ({len(ts_pairs)}): {ts_pairs_str}")
                print(f"\t- MODEL PREFIX: {run_prefix}")
                print(f"\t- LOSS MONITOR: {monitor}")

                # Train model
                t_model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)
                t_model.regularization_fn = lambda model, loss: regularization_fn(model, loss, d_tasks=d_tasks, reg_type=reg_type)

                # if checkpoint_path:
                #     print(f"\t- Loading previous checkpoint: {checkpoint_path}")
                #     model_state_dict = torch.load(checkpoint_path)
                #     model_state_dict = model_state_dict.get("state_dict", model_state_dict)
                #     t_model.load_state_dict(model_state_dict)

                # Set toolkit
                trainer = AutonmtTranslator(model=t_model, src_vocab=src_vocab, trg_vocab=trg_vocab,
                                           filter_tr_data_fn=_gen_filter_data_fn("train", valid_pairs=new_tr_pairs, past_pairs=past_pairs, ratio_past_data=ratio_past_data),
                                           filter_vl_data_fn=[_gen_filter_data_fn("val", valid_pairs=p) for p in ts_pairs],
                                           filter_ts_data_fn=[_gen_filter_data_fn("test", valid_pairs=p) for p in ts_pairs],
                                           )

                # Train model
                BATCH_SIZE = 64
                NUM_WORKERS = 10
                trainer.fit(default_ds, max_epochs=1, learning_rate=0.001, optimizer="adamw", gradient_clip_val=0.0,
                          monitor=monitor,batch_size=BATCH_SIZE, seed=1234, patience=10, num_workers=NUM_WORKERS,
                          devices="auto", accelerator="auto", strategy="ddp",
                          print_samples=3, wandb_params=wandb_params)
                asd = 333
                # Predict
                # model.load_best_checkpoint(model_ds=default_ds)
                # m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[1],
                #                          batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                #                          load_best_checkpoint=False, max_len_a=0.0, max_len_b=200,
                #                          model_ds=default_ds, force_overwrite=True)  # model_ds=train_ds => if fit() was not used before
                # scores.append(m_scores)
                ############################################################

                # Load model
                # checkpoint_path = model.load_best_checkpoint(model_ds=default_ds)
                #
                # # Compute weights and gradients
                # model.preprocess(default_ds, apply2train=True, apply2val=False, apply2test=False, force_overwrite=False)
                # weights, grads = compute_grads(model.model, model.train_tds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
                # d_tasks[run_prefix] = {"weights": weights, "gradients": grads}
                #
                # # Save weights and grads
                # run_name = default_ds.get_run_name(model.run_prefix)
                # path = default_ds.get_model_checkpoints_path(model.engine, run_name)
                # torch.save(weights, os.path.join(path, f'checkpoint_last__weights.pt'))
                # torch.save(grads, os.path.join(path, f'checkpoint_last__gradients.pt'))
                # torch.save(t_model.state_dict(), os.path.join(path, "checkpoint_last.pt"))
                # ############################################################

                # Add new pairs to past pairs
                if sequential_tr:
                    past_pairs.append(new_tr_pairs)
                else:
                    checkpoint_path = None

    print(f"Total models trained: {counter}")

    # # Make report
    # output_path = os.path.join(BASE_PATH, f".outputs/autonmt/{str(datetime.datetime.now())}")
    # df_report, df_summary = generate_report(scores=scores, output_path=output_path)
    #
    # # Print summary
    # print("Summary:")
    # print(df_summary.to_string(index=False))
    #
    # # Plot metrics
    # plots_path = os.path.join(output_path, "plots")
    # plot_metrics(output_path=plots_path, df_report=df_report, plot_metric="translations.beam1.sacrebleu_bleu_score",
    #              xlabel="MT Models", ylabel="BLEU Score", title="Model comparison")

if __name__ == "__main__":
    main()
    print("Done!")
