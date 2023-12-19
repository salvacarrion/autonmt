import datetime
import time
import os
import re
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


def add_prefix(data, ds):
    prefix = f"<{ds.src_lang}>-<{ds.trg_lang}>|"

    # Check if the data starts with the prefix
    if not bool(re.match(r"^<..>-<..>\|", data["lines"][0])):
        return [f"{prefix}{l}" for l in data["lines"]]
    else:
        return data["lines"]


def preprocess_predict(data, ds):
    if data["lang"] == ds.src_lang:  # Source
        return add_prefix(data, ds)
    else:  # Target
        return data["lines"]


def filter_data_fn(x, y, split_name, valid_pairs, past_pairs, ratio_past_data, from_fn=None, **kwargs):
    if not valid_pairs:  # Add all
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
    return fn_name, (lambda x, y, split_name=split_name, valid_pairs=valid_pairs, past_pairs=past_pairs, ratio_past_data=ratio_past_data, **kwargs: filter_data_fn(x, y, split_name, valid_pairs, past_pairs, ratio_past_data, **kwargs))


def compute_fisher_matrix(trainer, train_ds, learning_rate, batch_size, max_tokens=None, criterion="cross_entropy", num_workers=0):
    print(f"=> Computing Fisher information matrix...")
    import tqdm
    from torch.utils.data import DataLoader
    import torch.optim as optim

    # Preprocess data (loads the data and filters it)
    trainer.preprocess(train_ds, apply2train=True, apply2val=False, apply2test=False, force_overwrite=False)

    # Prepare dataloader
    train_loader = DataLoader(trainer.train_tds,
                              collate_fn=lambda x: trainer.train_tds.collate_fn(x, max_tokens=max_tokens),
                              num_workers=num_workers, pin_memory=True,
                              batch_size=batch_size, shuffle=False,
                              )

    # Set model to evaluation mode to avoid updating batch norm stats
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = trainer.model.to(device)
    model.eval()

    # Initialize Fisher information matrix as a dictionary of zeros with the same shape as the model's parameters
    fisher_dict = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
    grads_dict = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}

    # We will accumulate the gradients over the dataset
    model.zero_grad()
    for inputs, targets in tqdm.tqdm(train_loader, total=len(train_loader)):
        # Move data to the specified device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward
        output = model.forward_encoder(inputs)
        output = model.forward_decoder(targets, output)  # (Batch, Length, Embedding size)

        # Compute loss
        output = output.transpose(1, 2)[:, :, :-1]  # Remove last index to match shape with 'y[1:]'
        targets = targets[:, 1:]  # Remove <sos>

        # Compute the loss
        loss = torch.nn.functional.cross_entropy(output, targets, reduction='mean')
        loss.backward()

        # Accumulate Fisher information
        for name, param in model.named_parameters():
            fisher_dict[name] += param.grad ** 2
            grads_dict[name] += param.grad

        model.zero_grad()

    # Normalize by the number of samples
    for name in fisher_dict.keys():
        fisher_dict[name] = fisher_dict[name].detach().cpu().numpy() / len(trainer.train_tds)
        grads_dict[name] = grads_dict[name].detach().cpu().numpy() / len(trainer.train_tds)

    return fisher_dict, grads_dict


def get_weights(trainer):
    model = trainer.model
    model.eval()

    # Extract weights and gradients
    weights_dict = {}
    for name, param in model.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy()

    return weights_dict, model.state_dict()


def compute_grads(trainer, train_ds, learning_rate, batch_size, max_tokens=None, criterion="cross_entropy", num_workers=0):
    print(f"=> Computing gradients...")
    import tqdm
    from torch.utils.data import DataLoader
    import torch.optim as optim

    # Preprocess data (loads the data and filters it)
    trainer.preprocess(train_ds, apply2train=True, apply2val=False, apply2test=False, force_overwrite=False)

    # Prepare dataloader
    train_loader = DataLoader(trainer.train_tds,
                              collate_fn=lambda x: trainer.train_tds.collate_fn(x, max_tokens=max_tokens),
                              num_workers=num_workers, pin_memory=True,
                              batch_size=batch_size, shuffle=False,
                              )

    # Prepare model, criterion and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = trainer.model.to(device)
    model.configure_criterion(criterion)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Reset model
    model.train()
    optimizer.zero_grad()

    # Accumulate gradients
    for x, y in tqdm.tqdm(train_loader, total=len(train_loader)):
        batch = x.to(device), y.to(device)
        loss, _ = model._step(batch, 0, log_prefix=None)
        loss.backward()

    # Extract weights and gradients
    weights_dict = {}
    grad_dict = {}
    for name, param in model.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy()
        grad_dict[name] = param.grad.detach().cpu().numpy()

    return weights_dict, grad_dict, model.state_dict()


def regularization_fn(model, loss, alpha, **kwargs):
    d_tasks = kwargs.get("d_tasks")
    reg_type = kwargs.get("reg_type")

    # Apply regularization
    if reg_type is None:
        pass
    elif reg_type == "l1":
        for task_id in d_tasks.keys():
            for name, param in model.named_parameters():
                old_weight = torch.tensor(d_tasks[task_id]["weights"][name]).to(param.device)
                loss += ((param - old_weight).abs()).sum() * alpha
    elif reg_type == "l2":
        for task_id in d_tasks.keys():
            for name, param in model.named_parameters():
                old_weight = torch.tensor(d_tasks[task_id]["weights"][name]).to(param.device)
                loss += ((param - old_weight).pow(2)).sum() * alpha
    elif reg_type == "ewc":
        for task_id in d_tasks.keys():
            for name, param in model.named_parameters():
                fisher = torch.tensor(d_tasks[task_id]["fisher"][name]).to(param.device)
                old_weight = torch.tensor(d_tasks[task_id]["weights"][name]).to(param.device)
                loss += (fisher * (param - old_weight).pow(2)).sum() * alpha
    else:
        raise ValueError(f"Unknown value '{reg_type}' for reg_type")


# Preprocess functions
normalize_fn = lambda x: normalize_lines(x, seq=[NFKC(), Strip(), Lowercase()])
preprocess_raw_fn = lambda data, ds: preprocess_pairs(data["src"]["lines"], data["trg"]["lines"], normalize_fn=normalize_fn, min_len=1, max_len=None, remove_duplicates=False, shuffle_lines=False)
preprocess_splits_fn = lambda data, ds: preprocess_pairs(add_prefix(data["src"], ds), data["trg"]["lines"], normalize_fn=normalize_fn, shuffle_lines=False)
preprocess_predict_fn = lambda data, ds: preprocess_lines(preprocess_predict(data, ds), normalize_fn=normalize_fn)

# BASE_PATH = "/Users/salvacarrion/Documents/Programming/datasets/translate"  # Local
BASE_PATH2 = "/home/scarrion/datasets/translate"  # Remote
BASE_PATH3 = "/app/data"  # Docker
# BASE_PATH3 = "/Users/salvacarrion/Documents/Programming/datasets/optimus"  # Docker
BASE_PATH = BASE_PATH2 if os.environ.get("DEBUG", 0) else BASE_PATH3


def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        # Root folder for datasets
        base_path=BASE_PATH,

        # Set of datasets, languages, training sizes to try
        datasets=[
            {"name": "europarl", "languages": ["en-xx"], "sizes": [("original", None)]},
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
    default_ds = tr_datasets[0]

    # CF vars
    filter_tr_pairs = [["fr"]]  # Training data
    filter_ts_pairs = [["es"], ["fr"]]  # For each model
    sequential_tr = True

    for ratio_past_data in [1.0]: #[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        for reg_type in ["ewc"]:
            # Tmp vars
            past_pairs = []
            d_tasks = {}

            #######################################
            # Load initial checkpoint
            run_prefix = "cf-ft__tr_[]->[es]__reg_(none)"
            ckp_base_path = os.path.join(BASE_PATH, f"europarl/en-xx/original/models/autonmt/runs/{run_prefix}_bpe+bytes_8000/checkpoints")
            state_dict_path = os.path.join(ckp_base_path, f'state_dict__from_best.pt')
            weights_dict = torch.load(os.path.join(ckp_base_path, f'weights_dict__from_best.pt'))
            grads_dict = torch.load(os.path.join(ckp_base_path, f'grad_dict__from_best.pt'))
            fisher_dict = torch.load(os.path.join(ckp_base_path, f'fisher_dict__from_best.pt'))
            d_tasks[run_prefix] = {"weights": weights_dict, "gradients": grads_dict, "fisher": fisher_dict}
            #######################################

            scores = []
            for i, new_tr_pairs in enumerate(filter_tr_pairs):
                _filter_ts_pairs = ["xx" if x is None else '+'.join(x) for x in filter_ts_pairs]

                # Prettify past task and new tasks
                _past_pairs = [p[0] if p else "xx" for p in past_pairs]
                tr_pairs_old_str = ','.join(_past_pairs) if past_pairs else ""
                tr_pairs_new_str = '+'.join(new_tr_pairs) if new_tr_pairs else "xx"
                ts_pairs_str = '|'.join(_filter_ts_pairs)

                # Set run prefix
                alias = "cf-ft"
                tr_pairs_str = f"tr_[{tr_pairs_old_str}]->[{tr_pairs_new_str}]"
                tr_pairs_str += f"+[{tr_pairs_old_str}]x[{ratio_past_data}]"  # if ratio_past_data else ""
                tr_pairs_str += f"__reg_" + (reg_type.lower().strip() if reg_type else "(none)")
                run_prefix = f"{alias}__{tr_pairs_str}"

                # Set data and stuff
                new_tr_pair_str = new_tr_pairs[0] if new_tr_pairs else "xx"  # Trick: Only one pair
                monitor = f'val_{new_tr_pair_str}_loss/dataloader_idx_{_filter_ts_pairs.index(new_tr_pair_str)}'

                # Print info
                print(f"=> Training model...")
                print(f"\t- TRAINING ({i + 1}/{len(filter_tr_pairs)}): {tr_pairs_str}")
                print(f"\t- TESTING ({len(filter_ts_pairs)}): {ts_pairs_str}")
                print(f"\t- MODEL PREFIX: {run_prefix}")
                print(f"\t- LOSS MONITOR: {monitor}")
                ########################################

                # Instantiate vocabs and model
                src_vocab = Vocabulary(max_tokens=350).build_from_ds(ds=default_ds, lang=default_ds.src_lang)
                trg_vocab = Vocabulary(max_tokens=350).build_from_ds(ds=default_ds, lang=default_ds.trg_lang)
                model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)
                model.regularization_fn = lambda model, loss: regularization_fn(model, loss, d_tasks=d_tasks, reg_type=reg_type, alpha=ratio_past_data)

                # Load previous state_dict
                if state_dict_path:
                    print(f"\t- Loading previous state_dict")
                    model.load_state_dict(torch.load(state_dict_path))

                # Define trainer
                runs_dir = default_ds.get_runs_path(toolkit="autonmt")
                run_name = default_ds.get_run_name(run_prefix=run_prefix)  # + f"__{int(time.time())}"
                trainer = AutonmtTranslator(model=model, src_vocab=src_vocab, trg_vocab=trg_vocab,
                                            runs_dir=runs_dir, run_name=run_name,
                                            filter_tr_data_fn=_gen_filter_data_fn("train", valid_pairs=new_tr_pairs, past_pairs=past_pairs, ratio_past_data=None),
                                            filter_vl_data_fn=[_gen_filter_data_fn("val", valid_pairs=p) for p in filter_ts_pairs],
                                            filter_ts_data_fn=[_gen_filter_data_fn("test", valid_pairs=p) for p in filter_ts_pairs],
                                            )

                ############################################################
                # Train model
                wandb_params = dict(project="continual-learning-multi", entity="salvacarrion", reinit=True)
                trainer.fit(default_ds, max_epochs=5, learning_rate=0.001, optimizer="adam", batch_size=256, seed=None, monitor=monitor,
                            patience=10, num_workers=0, accelerator="auto", strategy="auto", save_best=True, save_last=True,
                            print_samples=1, wandb_params=wandb_params)
                ############################################################

                ############################################################
                # Test model
                # m_scores = trainer.predict(ts_datasets, metrics={"bleu"}, beams=[1], load_checkpoint="best",
                #                            preprocess_fn=preprocess_predict_fn, eval_mode="all", force_overwrite=False)
                #
                # # Add extra metrics
                # for ms in m_scores:
                #     ms['train_dataset'] = default_ds.dataset_name
                #     ms['vocab__merged'] = default_ds.merge_vocabs
                #     ms['train__lang_pair'] = f"[{tr_pairs_old_str}]->[{tr_pairs_new_str}]"
                #     ms['test__lang_pair'] = "Invalid"
                # scores.append(m_scores)
                ############################################################

                ############################################################
                # Compute fisher matrix
                # Load best model
                checkpoints_path = trainer.get_model_checkpoints_path()
                checkpoint_path__best = trainer._get_checkpoints(checkpoints_path, mode="best")
                trainer.load_checkpoint(checkpoint_path__best)

                # Save weights and state
                weights_dict, state_dict = get_weights(trainer)
                torch.save(weights_dict, os.path.join(checkpoints_path, f'weights_dict__from_best.pt'))
                torch.save(state_dict, os.path.join(checkpoints_path, f'state_dict__from_best.pt'))

                # Compute fisher matrix
                fisher_dict, grads_dict = compute_fisher_matrix(trainer, default_ds, learning_rate=0.001, batch_size=256)
                torch.save(fisher_dict, os.path.join(checkpoints_path, f'fisher_dict__from_best.pt'))
                torch.save(grads_dict, os.path.join(checkpoints_path, f'grads_dict__from_best.pt'))
                # ############################################################

                ############################################################
                # Add new pairs to past pairs
                if sequential_tr:
                    past_pairs.append(new_tr_pairs)
                    state_dict_path = os.path.join(checkpoints_path, f'state_dict__from_best.pt')
                    d_tasks[run_prefix] = {"weights": weights_dict, "gradients": grads_dict, "fisher": fisher_dict}
                else:
                    state_dict_path = None
                ############################################################

    # Make report
    output_path = os.path.join(BASE_PATH, f".outputs/autonmt/cf_ft__reg_l1-l2-ewc")
    df_report, df_summary = generate_report(scores=scores, output_path=output_path)

    # Print summary
    print("Summary:")
    print(df_summary.to_string(index=False))

    # # Plot metrics
    # plots_path = os.path.join(output_path, "plots")
    # plot_metrics(output_path=plots_path, df_report=df_report, plot_metric="translations.beam1.sacrebleu_bleu_score",
    #              xlabel="MT Models", ylabel="BLEU Score", title="Model comparison")
    print("DONEEE!!!")

if __name__ == "__main__":
    main()
