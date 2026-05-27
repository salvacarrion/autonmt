from abc import abstractmethod
import math
from collections import defaultdict

import pytorch_lightning as pl
import torch
from torch import nn

from autonmt.evaluation.metrics import score_sacrebleu
from autonmt.datasets.processors import decode_lines

from autonmt.utils.logger import get_logger

log = get_logger(__name__)


_OPTIMIZERS = {
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
    "sparseadam": torch.optim.SparseAdam,
}


class LitSeq2Seq(pl.LightningModule):

    def __init__(self, src_vocab_size, trg_vocab_size, padding_idx, packed_sequence=False, architecture=None, **kwargs):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.padding_idx = padding_idx
        self.packed_sequence = packed_sequence  # Use for RNNs and to "sort within batches"
        self.architecture = architecture if architecture else self.__class__.__name__

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

    @classmethod
    def from_vocabs(cls, src_vocab, trg_vocab, **kwargs):
        """Build the model inferring sizes / pad id from the vocabularies.

        Equivalent to:
            cls(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab),
                padding_idx=src_vocab.pad_id, **kwargs)

        ``src_vocab`` and ``trg_vocab`` must share ``pad_id`` (true by default
        for AutoNMT vocabularies); otherwise pass ``padding_idx`` explicitly.
        """
        assert src_vocab.pad_id == trg_vocab.pad_id, (
            f"src/trg vocabularies have different pad_id "
            f"({src_vocab.pad_id} vs {trg_vocab.pad_id}); "
            f"pass padding_idx= explicitly to the model constructor instead."
        )
        kwargs.setdefault("padding_idx", src_vocab.pad_id)
        return cls(
            src_vocab_size=len(src_vocab),
            trg_vocab_size=len(trg_vocab),
            **kwargs,
        )

    @abstractmethod
    def forward_encoder(self, x, x_len, **kwargs):
        pass

    @abstractmethod
    def forward_decoder(self, y, y_len, states, **kwargs):
        pass

    @abstractmethod
    def forward_enc_dec(self, x, x_len, y, y_len, **kwargs):
        pass

    def count_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        no_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = trainable_params + no_trainable_params
        return total_params, trainable_params, no_trainable_params

    def configure_optimizers(self):
        if not isinstance(self.optimizer, str):
            return self.optimizer
        key = self.optimizer.lower().strip()
        if key not in _OPTIMIZERS:
            raise ValueError(f"Unknown value '{self.optimizer}' for optimizer")
        return _OPTIMIZERS[key](self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def configure_criterion(self, criterion):
        if isinstance(criterion, str):
            key = criterion.lower().strip()
            if key != "cross_entropy":
                raise ValueError(f"Unknown value '{criterion}' for criterion")
            self.criterion_fn = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
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
        if self._print_samples:
            self._print_validation_samples()
        self.validation_step_outputs.clear()

    def _print_validation_samples(self):
        for (dl_idx, outputs), (fn_name, _) in zip(self.validation_step_outputs.items(), self._filter_eval):
            extra_info = f" (Filter: {fn_name}; val_dataloader_idx={dl_idx})" if dl_idx else ""
            log.info(f"=> Printing samples:" + extra_info)

            # Unpack per-batch outputs into flat lists per key
            d = defaultdict(list)
            for d_batch in outputs:
                for key, value in d_batch.items():
                    d[key].extend(value)

            samples = list(zip(d["src"], d["hyp"], d["ref"]))[:self._print_samples]
            for i, (src_i, hyp_i, ref_i) in enumerate(samples, 1):
                log.info(f"- Src. #{i}: {src_i}")
                log.info(f"- Ref. #{i}: {ref_i}")
                log.info(f"- Hyp. #{i}: {hyp_i}")
                log.info("")
            log.info("-" * 100)

    def _step(self, batch, batch_idx, log_prefix):
        (x, y), (x_len, y_len) = batch

        # Forward => (Batch, Length) => (Batch, Length, Vocab)
        # The input of the decoder needs the <sos>, but its output is shifted as it starts with the first word, not
        # with the <sos>. Therefore, we need to remove the last token from 'y'
        output = self.forward_enc_dec(x=x, x_len=x_len, y=y[:, :-1], y_len=y_len)

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

        outputs = None
        if log_prefix:
            sync_dist = (self.strategy == "ddp")

            try:
                ppl = math.exp(loss.item())
            except OverflowError:
                ppl = float("inf")
                log.warning("=> Overflow detected when computing perplexity. Set to 'inf'")

            self.log(f"{log_prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
            self.log(f"{log_prefix}_ppl", ppl, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
            self.log(f"{log_prefix}_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)

            if log_prefix.startswith("val"):
                outputs = self._compute_metrics(y_hat=predictions, y=y, x=x, log_prefix=log_prefix)
        return loss, outputs

    def _compute_metrics(self, y_hat, y, x, log_prefix):
        # Decode lines (only during training)
        # Since ref lines are encoded, unknowns can appear. Therefore, for small vocabularies the scores could be strongly biased
        src_vocab, trg_vocab = self._src_vocab, self._trg_vocab
        hyp_lines = [trg_vocab.decode(list(row)) for row in y_hat.detach().cpu().numpy()]
        ref_lines = [trg_vocab.decode(list(row)) for row in y.detach().cpu().numpy()]
        src_lines = [src_vocab.decode(list(row)) for row in x.detach().cpu().numpy()]

        hyp_lines = decode_lines(hyp_lines, trg_vocab.lang, trg_vocab.subword_model, trg_vocab.pretok_flag, trg_vocab.spm_model)
        ref_lines = decode_lines(ref_lines, trg_vocab.lang, trg_vocab.subword_model, trg_vocab.pretok_flag, trg_vocab.spm_model)
        src_lines = decode_lines(src_lines, src_vocab.lang, src_vocab.subword_model, src_vocab.pretok_flag, src_vocab.spm_model)

        scores = score_sacrebleu(hyp_lines=hyp_lines, ref_lines=ref_lines, metrics={"bleu"})

        sync_dist = (self.strategy == "ddp")
        for score in scores:
            metric_name = score['name'].lower()
            metric_key = f"{log_prefix}_{metric_name}"
            metric_key_best = f"{metric_key}_best"
            self.best_scores[metric_key] = max(score['score'], self.best_scores[metric_key])

            self.log(metric_key, score['score'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
            self.log(metric_key_best, self.best_scores[metric_key], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)

        return {"hyp": hyp_lines, "ref": ref_lines, "src": src_lines}
