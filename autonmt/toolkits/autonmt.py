import pytorch_lightning as pl
import torch
import wandb
import glob

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


class AutonmtTranslator(BaseTranslator):  # AutoNMT Translator

    def __init__(self, model, wandb_params=None, print_samples=False, **kwargs):
        super().__init__(engine="autonmt", **kwargs)
        self.model = model
        self.ckpt_cb = None
        self.wandb_params = wandb_params
        self.print_samples = print_samples

        # Translation preprocessing (do not confuse with 'train_ds')
        self.train_tds = None
        self.val_tds = None
        self.test_tds = None

    def _preprocess(self, ds, output_path, src_lang, trg_lang, train_path, val_path, test_path,
                    src_vocab_path, trg_vocab_path, force_overwrite, **kwargs):
        # Create preprocessing
        self.subword_model = ds.subword_model
        self.pretok_flag = ds.pretok_flag
        self.src_vocab_path = src_vocab_path
        self.trg_vocab_path = trg_vocab_path

        # Set common params
        params = dict(src_lang=src_lang, trg_lang=trg_lang, src_vocab=self.src_vocab, trg_vocab=self.trg_vocab,
                      filter_langs=ds.filter_train_langs)
        if not kwargs.get("external_data"):  # Training
            self.train_tds = Seq2SeqDataset(file_prefix=train_path, **params, **kwargs)
            self.val_tds = Seq2SeqDataset(file_prefix=val_path, **params, **kwargs)
        else:  # Evaluation
            self.test_tds = Seq2SeqDataset(file_prefix=test_path, **params, **kwargs)

    def _train(self, train_ds, checkpoints_dir, logs_path, max_tokens, batch_size, monitor, run_name,
               num_workers, patience, resume_training, force_overwrite, **kwargs):
        # Notes:
        # - "force_overwrite" is not needed. checkpoints are versioned, not deleted
        # - "resume_training" is not needed. models are initialized by the user.

        # Define dataloaders
        train_loader = DataLoader(self.train_tds, shuffle=True, collate_fn=lambda x: self.train_tds.collate_fn(x, max_tokens=max_tokens), batch_size=batch_size, num_workers=num_workers)
        val_loader = DataLoader(self.val_tds, shuffle=False, collate_fn=lambda x: self.val_tds.collate_fn(x, max_tokens=max_tokens), batch_size=batch_size, num_workers=num_workers)

        # Additional information for metrics
        self.model._src_vocab = self.train_tds.src_vocab
        self.model._trg_vocab = self.train_tds.trg_vocab
        self.model._subword_model = self.subword_model
        self.model._pretok_flag = self.pretok_flag
        self.model._src_model_vocab_path = self.src_vocab_path
        self.model._trg_model_vocab_path = self.trg_vocab_path
        self.model._print_samples = self.print_samples

        # Callbacks: Checkpoint
        callbacks = []
        mode = "min" if monitor == "loss" else "max"
        str_monitor = f"val_{monitor}"
        filename = "checkpoint_best__{epoch}-{" + str_monitor + ":.2f}"
        self.ckpt_cb = ModelCheckpoint(dirpath=checkpoints_dir, save_top_k=1, monitor=str_monitor, mode=mode,
                                       filename=filename, save_weights_only=True)
        self.ckpt_cb.FILE_EXTENSION = ".pt"
        callbacks += [self.ckpt_cb]

        # Callback: EarlyStop
        if patience:
            early_stop = EarlyStopping(monitor=str_monitor, patience=patience, mode=mode)
            callbacks += [early_stop]

        # Loggers
        tb_logger = TensorBoardLogger(save_dir=logs_path, name=run_name)
        loggers = [tb_logger]

        # Add wandb logger (if requested)
        if self.wandb_params:
            alias = f"{'_'.join(train_ds.id())}_{run_name}"
            wandb_logger = WandbLogger(name=alias, **self.wandb_params)
            loggers.append(wandb_logger)

            # Monitor
            wandb_logger.watch(self.model)

        # Training
        remove_params = {'weight_decay', 'criterion', 'optimizer', 'patience', 'seed', 'learning_rate', "fairseq_args"}
        pl_params = {k: v for k, v in kwargs.items() if k not in remove_params}
        trainer = pl.Trainer(logger=loggers, callbacks=callbacks, **pl_params)  # pl_params must be compatible with PL
        trainer.fit(self.model, train_loader, val_loader)

        # Force finish
        if self.wandb_params:
            wandb.finish()

    def _translate(self, model_ds, data_path, output_path, src_lang, trg_lang, beam_width, max_len_a, max_len_b, batch_size, max_tokens,
                   load_best_checkpoint, num_workers, devices, accelerator,
                   force_overwrite, checkpoints_dir=None, **kwargs):
        # Checkpoint
        if load_best_checkpoint:
            if self.ckpt_cb:
                checkpoint_path = self.ckpt_cb.best_model_path
            else:
                # Find last checkpoint
                checkpoints = sorted(glob.iglob(os.path.join(checkpoints_dir, "*.pt")), key=os.path.getctime, reverse=True)
                if not checkpoints:
                    raise ValueError(f"No checkpoints were found in {checkpoints_dir}")
                elif len(checkpoints) > 1:
                    checkpoint_path = checkpoints[0]
                    print(f"[WARNING] Multiple checkpoints were found. Using latest: {checkpoint_path}")
                else:
                    checkpoint_path = checkpoints[0]
                    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

            # Load checkpoint
            model_state_dict = torch.load(checkpoint_path)
            model_state_dict = model_state_dict.get("state_dict", model_state_dict)
            self.model.load_state_dict(model_state_dict)

        # Set evaluation model
        if accelerator in {"auto", "cuda", "gpu"} and self.model.device.type != "cuda":
            print(f"\t-[INFO]: Setting 'cuda' as the model's device")
            self.model = self.model.cuda()

        # Iterative decoding
        search_algorithm = beam_search if beam_width > 1 else greedy_search
        predictions, log_probabilities = search_algorithm(model=self.model, dataset=self.test_tds,
                                                          sos_id=self.test_tds.src_vocab.sos_id,
                                                          eos_id=self.test_tds.src_vocab.eos_id,
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
        hyp_tok = [self.test_tds.trg_vocab.decode(tokens) for tokens in predictions]
        write_file_lines(lines=hyp_tok, filename=os.path.join(output_path, "hyp.tok"), insert_break_line=True)

    @staticmethod
    def _count_model_parameters(model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        return trainable_params, non_trainable_params


