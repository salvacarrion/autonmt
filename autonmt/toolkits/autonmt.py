import os.path

import pytorch_lightning as pl
import torch
import wandb
import glob
import inspect

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from autonmt.bundle.utils import *
from autonmt.modules.datasets.seq2seq_dataset import Seq2SeqDataset
from autonmt.search.beam_search import beam_search
from autonmt.search.greedy_search import greedy_search
from autonmt.toolkits.base import BaseTranslator

from torch.utils.data.sampler import SequentialSampler
# from torchnlp.samplers import BucketBatchSampler

class AutonmtTranslator(BaseTranslator):  # AutoNMT Translator

    def __init__(self, model, **kwargs):
        super().__init__(engine="autonmt", **kwargs)
        self.model = model

        # Translation preprocessing (do not confuse with 'train_ds')
        self.train_tds = None
        self.val_tds = None
        self.test_tds = None


    def _preprocess(self, train_path, val_path, test_path,
                    apply2train, apply2val, apply2test,
                    src_lang, trg_lang, src_vocab_path, trg_vocab_path,
                    output_path, force_overwrite, **kwargs):

        # Set common params
        params = dict(src_lang=src_lang, trg_lang=trg_lang, src_vocab=self.src_vocab, trg_vocab=self.trg_vocab)

        # Training data
        if apply2train:
            fn_name, filter_fn = self.filter_tr_data_fn
            self.train_tds = Seq2SeqDataset(file_prefix=train_path, filter_fn=filter_fn, **params, **kwargs)

        # Validation data
        if apply2val:
            self.val_tds = []
            for fn_name, filter_fn in self.filter_vl_data_fn:
                sds = Seq2SeqDataset(file_prefix=val_path, filter_fn=filter_fn, **params, **kwargs)
                self.val_tds.append(sds)

        # Test data
        if apply2test:
            self.test_tds = []
            for fn_name, filter_fn in self.filter_ts_data_fn:
                sds = Seq2SeqDataset(file_prefix=test_path, filter_fn=filter_fn, **params, **kwargs)
                self.test_tds.append(sds)

    # def _len_func(self, ds, i):
    #     return len(ds.datasets.iloc[i]["src"].split())

    def _train(self, train_ds, checkpoints_dir, logs_path, force_overwrite, **kwargs):
        # Training params
        batch_size = kwargs.get("batch_size")
        max_tokens = kwargs.get("max_tokens")
        num_workers = kwargs.get("num_workers")
        monitor = kwargs.get("monitor")
        patience = kwargs.get("patience")
        save_last = kwargs.get("save_last")
        save_best = kwargs.get("save_best")
        wandb_params = kwargs.get("wandb_params")
        print_samples = kwargs.get("print_samples")
        skip_val_metrics = kwargs.get("skip_val_metrics")
        mode_str = "min" if "loss" in monitor.lower() else "max"
        ckpt_filename = "{epoch:03d}-{" + monitor.replace('/', '-') + ":.3f}"
        pin_memory = False if kwargs.get('devices') == "cpu" else True
        loggers, callbacks = [], []

        # Model hyperparams
        self.model.optimizer = kwargs.get("optimizer")
        self.model.learning_rate = kwargs.get("learning_rate")
        self.model.weight_decay = kwargs.get("weight_decay")
        self.model.configure_criterion(kwargs.get("criterion"))

        # Additional information for metrics
        self.model._src_vocab = self.train_tds.src_vocab
        self.model._trg_vocab = self.train_tds.trg_vocab
        self.model._filter_train = self.filter_tr_data_fn
        self.model._filter_eval = self.filter_vl_data_fn
        self.model._print_samples = print_samples
        self.model._skip_val_metrics = skip_val_metrics

        # Dataloader: Training
        train_loader = DataLoader(self.train_tds,
                                  collate_fn=lambda x: self.train_tds.collate_fn(x, max_tokens=max_tokens),
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  batch_size=batch_size, shuffle=True,
                                  )

        # Dataloader: Validation
        val_loaders = []
        for val_tds_i in self.val_tds:
            val_loaders.append(DataLoader(val_tds_i, shuffle=False,
                                          collate_fn=lambda x: val_tds_i.collate_fn(x, max_tokens=max_tokens),
                                          batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory))


        # Callbacks: Checkpoint
        ckpt_p = {}
        if save_best:
            ckpt_p.update({"monitor": monitor, "mode": mode_str, "filename": ckpt_filename + "__best"})
        if save_last:
            ckpt_p.update({"save_last": save_last})
        if ckpt_p:
            checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_dir, save_top_k=1, **ckpt_p)
            checkpoint_callback.FILE_EXTENSION = ".pt"
            if save_last:  # Change last checkpoint name
                checkpoint_callback.CHECKPOINT_NAME_LAST = ckpt_filename + "__last"
            callbacks += [checkpoint_callback]

        # Callback: EarlyStop
        if patience:
            early_stop = EarlyStopping(monitor=monitor, patience=patience, mode=mode_str)
            callbacks += [early_stop]

        # Loggers: Tensorboard
        if logs_path:
            tb_logger = TensorBoardLogger(save_dir=logs_path, name=self.run_name)
            loggers += [tb_logger]

        # Loggers: WandB
        if wandb_params:
            wandb_logger = WandbLogger(name=self.run_name, **wandb_params)
            loggers += [wandb_logger]

        # Training
        pl_whitelist = set(inspect.signature(pl.Trainer.__init__).parameters)
        pl_params = {k: v for k, v in kwargs.items() if k in pl_whitelist}
        trainer = pl.Trainer(logger=loggers, callbacks=callbacks, **pl_params)  # pl_params must be compatible with PL
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loaders)

        # Close stuff
        wandb.finish() if wandb_params else None


    def _translate(self, data_path, output_path, src_lang, trg_lang, beam_width, max_len_a, max_len_b, batch_size, max_tokens,
                   checkpoint, num_workers, devices, accelerator,
                   force_overwrite, checkpoints_dir=None, filter_idx=0, **kwargs):
        # Checkpoint
        if checkpoint:
            self.from_checkpoint = self.load_checkpoint(checkpoint)

        # Set evaluation model
        if accelerator in {"auto", "cuda", "gpu"} and self.model.device.type != "cuda":
            print(f"\t-[INFO]: Setting 'cuda' as the model's device")
            self.model = self.model.cuda()

        # Iterative decoding
        search_algorithm = beam_search if beam_width > 1 else greedy_search
        predictions, log_probabilities = search_algorithm(model=self.model, dataset=self.test_tds[filter_idx],
                                                          sos_id=self.trg_vocab.sos_id,
                                                          eos_id=self.trg_vocab.eos_id,
                                                          batch_size=batch_size, max_tokens=max_tokens,
                                                          beam_width=beam_width, max_len_a=max_len_a, max_len_b=max_len_b,
                                                          num_workers=num_workers)
        # Decode output
        self._postprocess_output(predictions=predictions, output_path=output_path)

    def _postprocess_output(self, predictions, output_path):
        """
        Important: src and ref will be overwritten with the original preprocessed files to avoid problems with unknowns
        """
        # Decode: hyp
        hyp_tok = [self.trg_vocab.decode(tokens) for tokens in predictions]
        write_file_lines(lines=hyp_tok, filename=os.path.join(output_path, "hyp.tok"), insert_break_line=True)

    @staticmethod
    def _count_model_parameters(model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        return trainable_params, non_trainable_params

    @staticmethod
    def _get_checkpoints(checkpoints_dir, mode):
        # Find checkpoint (sorted by creation time)
        checkpoint_paths = sorted([os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith(f'__{mode}.pt')], key=os.path.getctime, reverse=True)
        if not checkpoint_paths:
            raise ValueError(f"No ({mode}) checkpoints were found in {checkpoints_dir}")
        elif len(checkpoint_paths) > 1:  # Choose latests
            checkpoint_path = checkpoint_paths[0]
            print(f"[WARNING] Multiple checkpoints were found. Using more recent '{mode}': {checkpoint_path}")
        else:
            checkpoint_path = checkpoint_paths[0]
            print(f"[INFO] Checkpoint found: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint):
        # Get checkpoint path
        if os.path.isfile(checkpoint): # Path
            checkpoint_path = checkpoint
        elif checkpoint in {"best", "last"}:  # Checkpoint name
            checkpoint_path = self._get_checkpoints(self.get_model_checkpoints_path(), mode=checkpoint)
        else:
            raise ValueError("'checkpoint' must be a filename or 'best' or 'last'")

        # Load checkpoint
        _model = torch.load(checkpoint_path)
        self.model.load_state_dict(_model.get("state_dict", _model))
        return checkpoint_path


