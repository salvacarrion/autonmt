from collections import defaultdict

import pytorch_lightning as pl
import torch
from torch import nn

from autonmt.api.py_cmd_api import _sacrebleu
from autonmt.preprocessing.processors import decode_lines


class LitSeq2Seq(pl.LightningModule):

    def __init__(self, src_vocab_size, trg_vocab_size, padding_idx,
                 criterion="cross_entropy", learning_rate=0.001, optimizer="adam", weight_decay=0, **kwargs):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        # Hyperparams
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.optimizer = optimizer
        self.weight_decay = weight_decay

        # Set criterion
        self.criterion_fn = self.configure_criterion(padding_idx)

        # Other
        self.save_hyperparameters()
        self.best_scores = defaultdict(float)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError("Unknown value for optimizer")
        return optimizer

    def configure_criterion(self, padding_idx):
        if self.criterion == "cross_entropy":
            criterion_fn = nn.CrossEntropyLoss(ignore_index=padding_idx)
        else:
            raise ValueError("Unknown value for optimizer")
        return criterion_fn

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, log_prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, log_prefix="val")

    def _step(self, batch, batch_idx, log_prefix):
        x, y = batch

        # Forward
        output = self.forward_encoder(x)
        output = self.forward_decoder(y, output)  # (B, L, E)

        # Compute loss
        output = output.transpose(1, 2)[:, :, :-1]  # Remove last index to match shape with 'y[1:]'
        y = y[:, 1:]  # Remove <sos>
        loss = self.criterion_fn(output, y)

        # Metrics: Accuracy
        predictions = output.detach().argmax(1)
        batch_errors = (predictions != y).sum().item()
        accuracy = 1 - (batch_errors / predictions.numel())

        # Log params
        self.log(f"{log_prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{log_prefix}_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Compute metrics for validaiton
        if log_prefix == "val":
            self._compute_metrics(y_hat=predictions, y=y, metrics={"bleu"}, log_prefix=log_prefix)
        return loss

    def _compute_metrics(self, y_hat, y, metrics, log_prefix):
        # Decode lines
        # Since ref lines are encoded, unknowns can appear. Therefore, for small vocabularies the scores could be strongly biased
        hyp_lines = [self._trg_vocab.decode(list(x)) for x in y_hat.detach().cpu().numpy()]
        ref_lines = [self._trg_vocab.decode(list(x)) for x in y.detach().cpu().numpy()]

        # Full decoding
        hyp_lines = decode_lines(hyp_lines, self._trg_vocab.lang, self._subword_model, self._pretok_flag, self._trg_model_vocab_path,  remove_unk_hyphen=True)
        ref_lines = decode_lines(ref_lines, self._trg_vocab.lang, self._subword_model, self._pretok_flag, self._trg_model_vocab_path,  remove_unk_hyphen=True)

        # Compute metrics
        scores = []

        # Compute sacrebleu
        scores += _sacrebleu(hyp_lines=hyp_lines, ref_lines=ref_lines, metrics=metrics)

        # Log metrics
        for score in scores:
            # Get score and keep best score
            score_name = score['name'].lower()
            best_score = max(score['score'], self.best_scores[score_name])
            self.best_scores[score_name] = best_score

            self.log(f"{log_prefix}_{score_name}", score['score'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{log_prefix}_best_{score_name}", best_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
