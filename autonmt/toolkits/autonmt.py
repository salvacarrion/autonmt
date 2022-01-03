from autonmt.modules.datasets.seq2seq_dataset import Seq2SeqDataset
from autonmt.vocabularies.base_vocab import BaseVocabulary
from autonmt.vocabularies.whitespace_vocab import Vocabulary
from autonmt.modules.seq2seq import LitSeq2Seq
from autonmt.toolkits.base import BaseTranslator
from autonmt.search.greedy_search import greedy_search
from autonmt.search.beam_search import beam_search
from autonmt.bundle.utils import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from typing import Type

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class AutonmtTranslator(BaseTranslator):  # AutoNMT Translator

    def __init__(self, model, **kwargs):
        super().__init__(engine="autonmt", **kwargs)
        self.model = model

        # Translation preprocessing (do not confuse with 'train_ds')
        self.train_tds = None
        self.val_tds = None
        self.test_tds = None

    def _preprocess(self, src_lang, trg_lang, output_path, train_path, val_path, test_path, subword_model, **kwargs):
        # Create preprocessing
        # Set common params
        params = dict(src_lang=src_lang, trg_lang=trg_lang, src_vocab=self.src_vocab, trg_vocab=self.trg_vocab)
        if not kwargs.get("external_data"):  # Training
            self.train_tds = Seq2SeqDataset(file_prefix=train_path, **params, **kwargs)
            self.val_tds = Seq2SeqDataset(file_prefix=val_path, **params, **kwargs)
        else:  # Evaluation
            self.test_tds = Seq2SeqDataset(file_prefix=test_path, **params, **kwargs)

    def _train(self, data_bin_path, checkpoints_path, logs_path, max_tokens, batch_size, monitor, run_name,
               num_workers, patience, **kwargs):
        # Define dataloaders
        train_loader = DataLoader(self.train_tds, shuffle=True, collate_fn=lambda x: self.train_tds.collate_fn(x, max_tokens=max_tokens), batch_size=batch_size, num_workers=num_workers)
        val_loader = DataLoader(self.val_tds, shuffle=False, collate_fn=lambda x: self.val_tds.collate_fn(x, max_tokens=max_tokens), batch_size=batch_size, num_workers=num_workers)

        # Loggers
        tb_logger = TensorBoardLogger(save_dir=logs_path, name=run_name)
        loggers = [tb_logger]

        # Callbacks: Checkpoint
        callbacks = []
        mode = "min" if monitor == "loss" else "max"
        ckpt_cb = ModelCheckpoint(dirpath=checkpoints_path, save_top_k=1, monitor=f"val_{monitor}", mode=mode,
                                  filename="checkpoint_best", save_weights_only=True)
        ckpt_cb.FILE_EXTENSION = ".pt"
        callbacks += [ckpt_cb]

        # Callback: EarlyStop
        if patience:
            early_stop = EarlyStopping(monitor=f"val_{monitor}", patience=patience, mode=mode)
            callbacks += [early_stop]

        # Training
        remove_params = {'weight_decay', 'criterion', 'optimizer', 'patience', 'seed', 'learning_rate'}
        pl_params = {k: v for k, v in kwargs.items() if k not in remove_params}
        trainer = pl.Trainer(logger=loggers, callbacks=callbacks, **pl_params)  # pl_params must be compatible with PL
        trainer.fit(self.model, train_loader, val_loader)

    def _translate(self, src_lang, trg_lang, beam_width, max_gen_length, batch_size, max_tokens,
                   data_bin_path, output_path, num_workers, **kwargs):
        # Set evaluation model
        self.model.eval()

        # Iterative decoding
        search_algorithm = beam_search if beam_width > 1 else greedy_search
        predictions, log_probabilities = search_algorithm(model=self.model, dataset=self.test_tds,
                                                          sos_id=self.test_tds.src_vocab.sos_id,
                                                          eos_id=self.test_tds.src_vocab.eos_id,
                                                          batch_size=batch_size, max_tokens=max_tokens,
                                                          beam_width=beam_width, max_gen_length=max_gen_length,
                                                          num_workers=num_workers)
        # Decode output
        self._postprocess_output(predictions=predictions, output_path=output_path)

    def _postprocess_output(self, predictions, output_path):
        # Decode: hyp
        hyp_tok = [self.test_tds.trg_vocab.decode(tokens) for tokens in predictions]

        # Decode: src
        src_tok = []
        for line in self.test_tds.src_lines:
            line_enc = self.test_tds.src_vocab.encode(line)
            line_dec = self.test_tds.src_vocab.decode(line_enc)
            src_tok.append(line_dec)

        # Decode: ref
        ref_tok = []
        for line in self.test_tds.trg_lines:
            line_enc = self.test_tds.trg_vocab.encode(line)
            line_dec = self.test_tds.trg_vocab.decode(line_enc)
            ref_tok.append(line_dec)

        # Write file: hyp, src, ref
        for lines, fname in [(hyp_tok, "hyp.tok"), (src_tok, "src.tok"), (ref_tok, "ref.tok")]:
            write_file_lines(lines=lines, filename=os.path.join(output_path, fname))

    @staticmethod
    def _count_model_parameters(model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        return trainable_params, non_trainable_params


