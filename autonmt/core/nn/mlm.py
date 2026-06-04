"""Lightning base for encoder-only masked language models.

Third sibling of :class:`~autonmt.core.nn.seq2seq.LitSeq2Seq` (encoder–decoder)
and :class:`~autonmt.core.nn.lm.LitLM` (decoder-only): all three subclass
:class:`~autonmt.core.nn.base.LitBase` for the optimizer/scheduler/criterion
plumbing. The MLM loop is **bidirectional** and has **no shift** — the model
sees a corrupted block and recovers the original tokens at the masked positions
only. Corruption + target selection is done by
:class:`~autonmt.core.data.mlm_dataset.MLMDataset`; here the loss is simply
cross-entropy over the supervised (non-ignored) positions.

Validation reports masked-token accuracy and (pseudo-)perplexity rather than
BLEU — there is no reference to score against.

References
----------
Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
"""
from abc import abstractmethod
from collections import defaultdict

import torch

from autonmt.core.nn.base import LitBase
from autonmt.utils.logger import get_logger

log = get_logger(__name__)


class LitMLM(LitBase):
    """Base class for encoder-only masked LMs. Subclasses implement :meth:`forward`."""

    def __init__(self, vocab_size, padding_idx, block_size=None, architecture=None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        # Doubles as the criterion's ignore_index: MLMDataset writes padding_idx
        # into the label at every non-masked position (see LitBase.configure_criterion).
        self.padding_idx = padding_idx
        self.block_size = block_size
        self.architecture = architecture if architecture else self.__class__.__name__

        self.save_hyperparameters()
        self.best_scores = defaultdict(float)

    @classmethod
    def from_corpus(cls, corpus, **kwargs):
        """Build the model inferring vocab size / pad id from an :class:`LMCorpus`.

        Equivalent to ``cls(vocab_size=corpus.model_vocab_size,
        padding_idx=corpus.pad_id, **kwargs)``. Use a ``mode="mlm"`` corpus.
        """
        kwargs.setdefault("padding_idx", corpus.pad_id)
        return cls(vocab_size=corpus.model_vocab_size, **kwargs)

    @abstractmethod
    def forward(self, x, attention_mask=None):
        """Map ``x`` ``(B, L)`` token ids to logits ``(B, L, vocab_size)``.

        Bidirectional: every position attends to the whole sequence (no causal
        mask). ``attention_mask`` (True = keep) is optional — packed MLM has no
        padding, so it is usually ``None``.
        """

    # --- Training loop --------------------------------------------------

    def _step(self, batch, log_prefix):
        x, y = batch                                  # (B, L), (B, L) with ignore at non-masked
        logits = self(x)                              # (B, L, V)
        loss = self.criterion_fn(logits.transpose(1, 2), y)   # ignores padding_idx targets

        if self.regularization_fn:
            self.regularization_fn(self, loss)

        sync_dist = (self.strategy == "ddp")
        with torch.no_grad():
            preds = logits.detach().argmax(-1)
            counted = (y != self.padding_idx)         # only the masked positions
            denom = counted.sum().clamp(min=1).float()
            accuracy = ((preds == y) & counted).sum().float() / denom
            ppl = torch.exp(loss.detach().clamp(max=20.0))

        self.log(f"{log_prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
        self.log(f"{log_prefix}_ppl", ppl, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
        self.log(f"{log_prefix}_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)

        if log_prefix.startswith("val"):
            key = f"{log_prefix}_loss_best"
            prev = self.best_scores.get(key)
            self.best_scores[key] = loss.item() if prev is None else min(prev, loss.item())
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        return self._step(batch, log_prefix="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self._step(batch, log_prefix="val")
