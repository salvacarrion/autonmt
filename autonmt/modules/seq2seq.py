from abc import ABC, abstractmethod
import math
from collections import defaultdict

import pytorch_lightning as pl
import torch
from torch import nn

from autonmt.bundle.metrics import _sacrebleu  # TODO: I don't like this
from autonmt.preprocessing.processors import decode_lines


class LitSeq2Seq(pl.LightningModule):

    def __init__(self, src_vocab_size, trg_vocab_size, padding_idx, **kwargs):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.padding_idx = padding_idx

        # Hyperparams (PyTorch Lightning stuff)
        self.strategy = None
        self.optimizer = None
        self.learning_rate = None
        self.weight_decay = None
        self.criterion_fn = None
        self.regularization_fn = None

        # Other
        self.save_hyperparameters()
        self.best_scores = defaultdict(float)
        self.validation_step_outputs = defaultdict(list)

    @abstractmethod
    def forward_encoder(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward_decoder(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        optim_fn = {
            "adadelta": torch.optim.Adadelta,
            "adagrad": torch.optim.Adagrad,
            "adam": torch.optim.Adam,
            "adamax": torch.optim.Adamax,
            "adamw": torch.optim.AdamW,
            "asgd": torch.optim.ASGD,
            "lbfgs": torch.optim.LBFGS,
            "nadam": torch.optim.NAdam,
            "radam": torch.optim.RAdam,
            "rmsprop": torch.optim.RMSprop,
            "rprop": torch.optim.Rprop,
            "sgd": torch.optim.SGD,
            "sparseadam": torch.optim.SparseAdam
        }

        # Select optimizer
        if isinstance(self.optimizer, str):
            optim_key = self.optimizer.lower().strip()
            if optim_key in optim_fn:
                return optim_fn[optim_key](self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            else:
                raise ValueError(f"Unknown value '{self.optimizer}' for optimizer")
        else:
            return self.optimizer

    def configure_criterion(self, criterion):
        # Set criterion
        if isinstance(criterion, str):
            criterion_key = criterion.lower().strip()
            if criterion_key == "cross_entropy":
                self.criterion_fn = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
            else:
                raise ValueError(f"Unknown value '{criterion}' for criterion")
        else:
            self.criterion_fn = criterion

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        loss, _ = self._step(batch, batch_idx, log_prefix=f"train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        eval_prefix = "val"
        if dataloader_idx is not None:
            fn_name, _ = self._filter_eval[dataloader_idx]
            eval_prefix += "_" + fn_name
        loss, outputs = self._step(batch, batch_idx, log_prefix=eval_prefix)
        self.validation_step_outputs[dataloader_idx].append(outputs)
        return loss, outputs

    def on_validation_epoch_end(self):
        # Print samples (if enabled)
        if self._print_samples:
            for (dl_idx, outputs), (fn_name, _) in zip(self.validation_step_outputs.items(), self._filter_eval):  # Iterate over dataloader
                extra_info = f" (Filter: {fn_name}; val_dataloader_idx={dl_idx})" if dl_idx else ""
                print(f"=> Printing samples:" + extra_info)

                # Unpack outputs
                d = defaultdict(list)
                for d_batch in outputs:
                    for key, value in d_batch.items():
                        d[key].extend(value)
                d = dict(d)

                # Print samples
                src, hyp, ref = d["src"], d["hyp"], d["ref"]
                for i, (src_i, hyp_i, ref_i) in enumerate(list(zip(src, hyp, ref))[:self._print_samples], 1):
                    print(f"- Src. #{i}: {src_i}")
                    print(f"- Ref. #{i}: {ref_i}")
                    print(f"- Hyp. #{i}: {hyp_i}")
                    print("")
                print("-"*100)

        # Free memory
        self.validation_step_outputs.clear()

    def forward_enc_dec(self, x, y):
        values = self.forward_encoder(x)
        values = self.forward_decoder(y, **values)  # (B, L, E)
        output = values["output"]
        return output

    def _step(self, batch, batch_idx, log_prefix):
        x, y = batch

        # Forward => (Batch, Length) => (Batch, Length, Vocab)
        # The input of the decoder needs the <sos>, but its output is shifted as it starts with the first word, not
        # with the <sos>. Therefore, we need to remove the last token from 'y'
        output = self.forward_enc_dec(x, y[:, :-1])

        # Remove the <sos> token from the target
        y = y[:, 1:]

        # Compute loss
        output = output.transpose(1, 2)  # (B, L, V) => (B, V, L)
        loss = self.criterion_fn(output, y)  # (B, V, L) vs (B, L)

        # Apply regularization
        if self.regularization_fn:
            self.regularization_fn(self, loss)

        # Metrics: Accuracy
        predictions = output.detach().argmax(1)
        batch_errors = (predictions != y).sum().item()
        accuracy = 1 - (batch_errors / predictions.numel())

        # Log params
        outputs = None
        if log_prefix:  # Not clear is 'sync_dist=True' should be enabled by default
            sync_dist = (self.strategy == "ddp")

            # Control for overflow (Should it simply fail?)
            try:
                ppl = math.exp(loss)
            except OverflowError as e:
                ppl = float("inf")
                print("=> [WARNING] Overflow detected when computing perplexity. Set to 'inf'")

            # Log metrics
            self.log(f"{log_prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
            self.log(f"{log_prefix}_ppl", ppl, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
            self.log(f"{log_prefix}_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)

            # Compute metrics for validation
            if log_prefix.startswith("val"):
                outputs = self._compute_metrics(y_hat=predictions, y=y, metrics={"bleu"}, x=x, log_prefix=log_prefix)
        return loss, outputs

    def _compute_metrics(self, y_hat, y, x, metrics, log_prefix):
        # Decode lines (only during training)
        # Since ref lines are encoded, unknowns can appear. Therefore, for small vocabularies the scores could be strongly biased
        hyp_lines = [self._trg_vocab.decode(list(x)) for x in y_hat.detach().cpu().numpy()]
        ref_lines = [self._trg_vocab.decode(list(x)) for x in y.detach().cpu().numpy()]
        src_lines = [self._src_vocab.decode(list(x)) for x in x.detach().cpu().numpy()]

        # Full decoding (lines are stripped)
        hyp_lines = decode_lines(hyp_lines, self._trg_vocab.lang, self._trg_vocab.subword_model, self._trg_vocab.pretok_flag, self._trg_vocab.spm_model)
        ref_lines = decode_lines(ref_lines, self._trg_vocab.lang, self._trg_vocab.subword_model, self._trg_vocab.pretok_flag, self._trg_vocab.spm_model)
        src_lines = decode_lines(src_lines, self._src_vocab.lang, self._src_vocab.subword_model, self._src_vocab.pretok_flag, self._src_vocab.spm_model)

        # Compute metrics
        scores = []

        # Compute sacrebleu
        scores += _sacrebleu(hyp_lines=hyp_lines, ref_lines=ref_lines, metrics=metrics)

        # Log metrics
        for score in scores:
            # Get score and keep best score
            metric_name = score['name'].lower()
            metric_key = f"{log_prefix}_{metric_name}"

            # Get best
            metric_key_best = f"{metric_key}_best"
            self.best_scores[metric_key] = max(score['score'], self.best_scores[metric_key])

            # Log metrics
            sync_dist = (self.strategy == "ddp")
            self.log(metric_key, score['score'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
            self.log(metric_key_best, self.best_scores[metric_key], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)

        return {"hyp": hyp_lines, "ref": ref_lines, "src": src_lines}
