"""Modality-agnostic Lightning base shared by every AutoNMT neural model.

``LitBase`` owns the plumbing that does not depend on *what* the model maps
(text→text, single-stream LM, …): the optimizer registry, optimizer/scheduler
construction, the criterion factory and parameter counting. The seq2seq-specific
training loop (teacher forcing, BLEU validation) lives in
:class:`~autonmt.core.nn.seq2seq.LitSeq2Seq`; the decoder-only LM loop lives in
:class:`~autonmt.core.nn.lm.LitLM`. Both subclass this.

Splitting this out (rather than duplicating the optimizer/scheduler code) keeps
a single source of truth for the noam / inverse-sqrt schedules and the optimizer
table, so the LM and seq2seq paths can never silently drift apart.
"""
import pytorch_lightning as pl
import torch
from torch import nn


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


class LitBase(pl.LightningModule):
    """Lightning base with the optimizer/scheduler/criterion plumbing.

    Subclasses set the hyperparameter attributes below (usually via the
    translator at train time) and implement their own ``training_step`` /
    ``validation_step``. ``configure_criterion`` references ``self.padding_idx``,
    so subclasses that use the built-in ``"cross_entropy"`` string must define
    that attribute.
    """

    def __init__(self):
        super().__init__()
        # Hyperparams (set by the translator before fit; see configure_optimizers).
        self.strategy = None
        self.optimizer = None
        self.learning_rate = None
        self.weight_decay = None
        self.scheduler = None
        self.warmup_steps = None
        self.criterion_fn = None
        self.regularization_fn = None

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
