from abc import abstractmethod
from collections import defaultdict

import pytorch_lightning as pl
import torch
from torch import nn

from autonmt.evaluation.metrics import score_sacrebleu
from autonmt.datasets.encoding import decode_lines

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

    def __init__(self, src_vocab_size, tgt_vocab_size, padding_idx, packed_sequence=False, architecture=None, **kwargs):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.padding_idx = padding_idx
        self.packed_sequence = packed_sequence  # Use for RNNs and to "sort within batches"
        self.architecture = architecture if architecture else self.__class__.__name__

        # Hyperparams (PyTorch Lightning stuff)
        self.strategy = None
        self.optimizer = None
        self.learning_rate = None
        self.weight_decay = None
        self.scheduler = None
        self.warmup_steps = None
        self.criterion_fn = None
        self.regularization_fn = None

        # Other
        self.save_hyperparameters()
        self.best_scores = defaultdict(float)
        self.validation_step_outputs = defaultdict(list)

    @classmethod
    def from_vocabs(cls, src_vocab, tgt_vocab, **kwargs):
        """Build the model inferring sizes / pad id from the vocabularies.

        Equivalent to:
            cls(src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
                padding_idx=src_vocab.pad_id, **kwargs)

        ``src_vocab`` and ``tgt_vocab`` must share ``pad_id`` (true by default
        for AutoNMT vocabularies); otherwise pass ``padding_idx`` explicitly.
        """
        assert src_vocab.pad_id == tgt_vocab.pad_id, (
            f"src/tgt vocabularies have different pad_id "
            f"({src_vocab.pad_id} vs {tgt_vocab.pad_id}); "
            f"pass padding_idx= explicitly to the model constructor instead."
        )
        kwargs.setdefault("padding_idx", src_vocab.pad_id)
        return cls(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
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
        if isinstance(self.optimizer, str):
            key = self.optimizer.lower().strip()
            if key not in _OPTIMIZERS:
                raise ValueError(f"Unknown value '{self.optimizer}' for optimizer")
            optimizer = _OPTIMIZERS[key](self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer

        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            # interval="step" so noam/inverse_sqrt update LR per optimizer step,
            # not per epoch — they're step-based by definition.
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _build_scheduler(self, optimizer):
        """Resolve ``self.scheduler`` into a torch LR scheduler.

        Accepts the string presets ``"noam"`` and ``"inverse_sqrt"``, a callable
        ``(optimizer) -> scheduler``, or an already-built scheduler instance.

        References
        ----------
        Vaswani et al. (2017). *Attention Is All You Need.* (noam schedule, §5.3)
        [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

        Ott et al. (2019). *fairseq: A Fast, Extensible Toolkit for Sequence
        Modeling.* (inverse-sqrt schedule)
        [arXiv:1904.01038](https://arxiv.org/abs/1904.01038)
        """
        s = self.scheduler
        if s is None:
            return None
        if isinstance(s, str):
            key = s.lower().strip()
            warmup = max(self.warmup_steps or 4000, 1)
            if key == "noam":
                # Vaswani et al. (2017) §5.3: factor peaks at 1.0 at step=warmup,
                # decays as step^-0.5 afterwards. Multiplies the optimizer's base lr.
                def lr_lambda(step):
                    step = max(step, 1)
                    return (warmup ** 0.5) * min(step ** -0.5, step * warmup ** -1.5)
                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            if key == "inverse_sqrt":
                # Fairseq default: linear warmup to 1.0, then 1/sqrt decay.
                def lr_lambda(step):
                    if step < warmup:
                        return step / warmup
                    return (warmup / step) ** 0.5
                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            raise ValueError(
                f"Unknown scheduler '{s}'. Use 'noam', 'inverse_sqrt', or pass a callable "
                f"that takes (optimizer) and returns a torch.optim.lr_scheduler."
            )
        if callable(s):
            return s(optimizer)
        return s  # assume already a torch lr_scheduler instance

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
        # Aggregate hyp/ref across batches and compute corpus-BLEU **once**.
        # Per-batch BLEU is both expensive (Moses + sacrebleu + CUDA syncs per
        # batch) and statistically wrong: corpus-BLEU is non-linear in the
        # underlying n-gram counts so averaging batch scores is not the corpus
        # score.
        sync_dist = (self.strategy == "ddp")
        for dl_idx, outputs in self.validation_step_outputs.items():
            if not outputs:
                continue
            hyp_lines, ref_lines = [], []
            for batch_out in outputs:
                hyp_lines.extend(batch_out["hyp"])
                ref_lines.extend(batch_out["ref"])
            if not hyp_lines:
                continue

            prefix = "val"
            if dl_idx is not None and self._filter_eval is not None:
                fn_name, _ = self._filter_eval[dl_idx]
                if fn_name:
                    prefix += "_" + fn_name

            scores = score_sacrebleu(hyp_lines=hyp_lines, ref_lines=ref_lines,
                                     metrics={"bleu"})
            for score in scores:
                metric_name = score['name'].lower()
                metric_key = f"{prefix}_{metric_name}"
                metric_key_best = f"{metric_key}_best"
                self.best_scores[metric_key] = max(score['score'], self.best_scores[metric_key])
                self.log(metric_key, score['score'],
                         on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
                self.log(metric_key_best, self.best_scores[metric_key],
                         on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)

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

        # Metrics: accuracy as a GPU tensor (Lightning defers the host sync
        # until log flush). Calling ``.item()`` here would block the stream
        # every training step and stall the next batch's prefetch.
        predictions = output.detach().argmax(1)
        correct = (predictions == y).sum().float()
        accuracy = correct / predictions.numel()

        outputs = None
        if log_prefix:
            sync_dist = (self.strategy == "ddp")

            # Clamp before exp so PPL stays finite on the device — matches the
            # spirit of the old OverflowError fallback without the sync.
            ppl = torch.exp(loss.detach().clamp(max=20.0))

            self.log(f"{log_prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
            self.log(f"{log_prefix}_ppl", ppl, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
            self.log(f"{log_prefix}_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)

            if log_prefix.startswith("val"):
                outputs = self._compute_metrics(y_hat=predictions, y=y, x=x, log_prefix=log_prefix)
        return loss, outputs

    def _compute_metrics(self, y_hat, y, x, log_prefix):
        # Decode hyp/ref/src to text and return for epoch-end aggregation.
        # Scoring (sacrebleu) is deferred to ``on_validation_epoch_end`` so
        # we compute corpus-BLEU once per epoch instead of once per batch.
        # Since ref lines are encoded, unknowns can appear. Therefore, for small vocabularies the scores could be strongly biased
        src_vocab, tgt_vocab = self._src_vocab, self._tgt_vocab
        hyp_lines = [tgt_vocab.decode(list(row)) for row in y_hat.detach().cpu().numpy()]
        ref_lines = [tgt_vocab.decode(list(row)) for row in y.detach().cpu().numpy()]
        src_lines = [src_vocab.decode(list(row)) for row in x.detach().cpu().numpy()]

        hyp_lines = decode_lines(hyp_lines, tgt_vocab.lang, tgt_vocab.subword_model, tgt_vocab.pretok_flag, tgt_vocab.spm_model)
        ref_lines = decode_lines(ref_lines, tgt_vocab.lang, tgt_vocab.subword_model, tgt_vocab.pretok_flag, tgt_vocab.spm_model)
        src_lines = decode_lines(src_lines, src_vocab.lang, src_vocab.subword_model, src_vocab.pretok_flag, src_vocab.spm_model)

        return {"hyp": hyp_lines, "ref": ref_lines, "src": src_lines}
