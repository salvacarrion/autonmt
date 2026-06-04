"""Lightning base for decoder-only language models.

Sibling of :class:`~autonmt.core.nn.seq2seq.LitSeq2Seq`: both subclass
:class:`~autonmt.core.nn.base.LitBase` for the optimizer/scheduler/criterion
plumbing, but the LM loop is a strict simplification — no encoder, no teacher
forcing of a separate target stream. The model maps ``x`` (a block of tokens)
to next-token logits; the loss is cross-entropy of ``logits`` vs ``x`` shifted
by one, restricted to the supervised positions.

The ``supervise`` mask (provided per-item by
:class:`~autonmt.core.data.lm_dataset.LMDataset`) is what makes pure-LM and
instruct training a single code path: it is all-ones for pure LM, and zero over
the prompt span for instruct corpora, so the prompt never contributes to the
loss.

Validation reports perplexity (intrinsic) rather than BLEU — there is no
reference translation to score against.

References
----------
Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.*
(decoder-only next-token language modelling)
[OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Ouyang et al. (2022). *Training Language Models to Follow Instructions with
Human Feedback.* (the prompt-masked, completion-only objective implemented here
for instruct corpora) [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
"""
from abc import abstractmethod
from collections import defaultdict

import torch

from autonmt.core.nn.base import LitBase
from autonmt.utils.logger import get_logger

log = get_logger(__name__)


class LitLM(LitBase):
    """Base class for decoder-only LMs. Subclasses implement :meth:`forward`."""

    def __init__(self, vocab_size, padding_idx, block_size=None, architecture=None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        # block_size is the training context length; informational here (the
        # positional bound lives on the concrete model, e.g. GPT.max_seq_len).
        self.block_size = block_size
        self.architecture = architecture if architecture else self.__class__.__name__

        self.save_hyperparameters()
        self.best_scores = defaultdict(float)

    @classmethod
    def from_corpus(cls, corpus, **kwargs):
        """Build the model inferring vocab size / pad id from an :class:`LMCorpus`.

        Equivalent to ``cls(vocab_size=corpus.model_vocab_size,
        padding_idx=corpus.pad_id, **kwargs)``.
        """
        kwargs.setdefault("padding_idx", corpus.pad_id)
        return cls(vocab_size=corpus.model_vocab_size, **kwargs)

    @abstractmethod
    def forward(self, x, incremental_state=None):
        """Map ``x`` ``(B, L)`` token ids to logits ``(B, L, vocab_size)``.

        When ``incremental_state`` is a dict, the model uses KV-cached decoding
        (one or more new tokens per call); when ``None`` it runs the full
        parallel forward used for training.
        """

    # --- Training loop --------------------------------------------------

    def _step(self, batch, log_prefix):
        x, y, supervise = batch                       # each (B, L)
        logits = self(x)                              # (B, L, V)

        # Mask out non-supervised target positions (prompt span for instruct,
        # nothing for pure LM) by setting them to the criterion's ignore_index.
        y = y.masked_fill(supervise == 0, self.padding_idx)
        loss = self.criterion_fn(logits.transpose(1, 2), y)   # (B, V, L) vs (B, L)

        if self.regularization_fn:
            self.regularization_fn(self, loss)

        sync_dist = (self.strategy == "ddp")
        with torch.no_grad():
            preds = logits.detach().argmax(-1)
            counted = (y != self.padding_idx)
            denom = counted.sum().clamp(min=1).float()
            accuracy = ((preds == y) & counted).sum().float() / denom
            # Clamp before exp so PPL stays finite on-device (matches LitSeq2Seq).
            ppl = torch.exp(loss.detach().clamp(max=20.0))

        self.log(f"{log_prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
        self.log(f"{log_prefix}_ppl", ppl, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
        self.log(f"{log_prefix}_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)

        # Track best (lowest) val loss for convenience / reporting.
        if log_prefix.startswith("val"):
            key = f"{log_prefix}_loss_best"
            prev = self.best_scores.get(key)
            self.best_scores[key] = loss.item() if prev is None else min(prev, loss.item())
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        return self._step(batch, log_prefix="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self._step(batch, log_prefix="val")
