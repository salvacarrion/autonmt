"""Trainer for encoder-only masked language models.

Subclass of :class:`~autonmt.backends.lm.trainer.LMTrainer` that swaps three
things and reuses everything else (config, run layout, the PyTorch-Lightning
``fit`` scaffolding):

  * :meth:`_make_datasets` builds :class:`MLMDataset` (dynamic masking) instead
    of the causal :class:`LMDataset`.
  * :meth:`evaluate` reports masked-token accuracy + pseudo-perplexity.
  * :meth:`fill_mask` replaces ``generate`` — an MLM doesn't decode left to
    right, it predicts the tokens at ``<mask>`` positions.

Use it with a ``mode="mlm"`` :class:`~autonmt.datasets.lm_corpus.LMCorpus`.

References
----------
Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
"""
import torch
from torch.utils.data import DataLoader

from autonmt.backends.lm.trainer import LMTrainer
from autonmt.core.data.mlm_dataset import MLMDataset
from autonmt.utils.logger import get_logger

log = get_logger(__name__)


class MLMTrainer(LMTrainer):
    """Train / evaluate / fill-mask an encoder-only masked LM on an :class:`LMCorpus`."""

    def __init__(self, model, mlm_prob=0.15, **kwargs):
        super().__init__(model=model, **kwargs)
        self.mlm_prob = mlm_prob

    def _make_datasets(self, corpus, block_size):
        if corpus.mask_id is None:
            raise ValueError(
                "MLMTrainer needs a corpus built with mode='mlm' (which reserves a "
                f"<mask> token); got mode={corpus.mode!r}."
            )
        # ignore_index = pad_id so the labels MLMDataset writes line up with the
        # criterion's ignore_index (LitBase builds CrossEntropyLoss(ignore_index=padding_idx)).
        params = dict(block_size=block_size, mask_id=corpus.mask_id,
                      vocab_size=corpus.model_vocab_size, special_ids=corpus.special_ids(),
                      mlm_prob=self.mlm_prob, ignore_index=corpus.pad_id)
        return (MLMDataset(corpus.tokens_file(corpus.train_name), **params),
                MLMDataset(corpus.tokens_file(corpus.val_name), **params))

    @torch.no_grad()
    def evaluate(self, corpus=None, split=None, block_size=256, batch_size=16, accelerator="auto"):
        """Masked-token accuracy + (pseudo-)perplexity on a split.

        Masking is dynamic, so the number varies slightly run-to-run.
        """
        corpus = corpus or self.corpus
        split = split or corpus.val_name
        device = self._resolve_device(accelerator)
        self.model = self.model.to(device).eval()

        ds = MLMDataset(corpus.tokens_file(split), block_size=block_size, mask_id=corpus.mask_id,
                        vocab_size=corpus.model_vocab_size, special_ids=corpus.special_ids(),
                        mlm_prob=self.mlm_prob, ignore_index=corpus.pad_id)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=corpus.pad_id, reduction="sum")

        total_loss, total_tokens, correct = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = self.model(x)
            total_loss += criterion(logits.transpose(1, 2), y).item()
            counted = (y != corpus.pad_id)
            total_tokens += int(counted.sum().item())
            correct += int(((logits.argmax(-1) == y) & counted).sum().item())

        total_tokens = max(total_tokens, 1)
        mean_nll = total_loss / total_tokens
        ppl = float(torch.exp(torch.tensor(mean_nll)))
        acc = correct / total_tokens
        log.info(f"=> [Evaluate MLM]: split={split!r} loss={mean_nll:.4f} ppl={ppl:.2f} "
                 f"masked_acc={acc:.4f} ({total_tokens} masked tokens)")
        return {"loss": mean_nll, "ppl": ppl, "masked_acc": acc, "tokens": total_tokens}

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "MLMTrainer does not do autoregressive generation. Use fill_mask(text) to "
            "predict the tokens at <mask> positions."
        )

    @torch.no_grad()
    def fill_mask(self, text, top_k=1, corpus=None, add_sos=True, accelerator="auto"):
        """Predict the token(s) at each ``<mask>`` in ``text``.

        Returns a list (one entry per masked position) of the top-``top_k``
        predicted pieces. Put the literal ``<mask>`` in the text, e.g.
        ``"the <mask> fox jumps"``. ``add_sos`` prepends the ``<s>`` the corpus
        wraps every document with, so the context matches what the model trained on.
        """
        corpus = corpus or self.corpus
        if corpus is None:
            raise ValueError("fill_mask() needs a corpus to (de)tokenize text.")
        device = self._resolve_device(accelerator)
        self.model = self.model.to(device).eval()

        # Splice mask_id at each literal "<mask>" and SentencePiece-encode the text
        # chunks around it. We don't rely on the tokenizer to extract "<mask>"
        # itself — user-symbol extraction is unreliable for some model types (e.g.
        # word), and this keeps the masked position a single clean token, matching
        # how MLMDataset corrupts during training.
        sp = corpus.load_spm()
        ids = [corpus.sos_id] if add_sos else []
        for i, chunk in enumerate(text.split(corpus.mask_piece)):
            if i > 0:
                ids.append(corpus.mask_id)
            ids.extend(sp.encode(chunk, out_type=int))
        if corpus.mask_id not in ids:
            log.warning(f"fill_mask: no {corpus.mask_piece} found in the input text.")
            return []

        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = self.model(x)[0]                       # (L, V)
        return [
            [corpus.decode([t]) for t in logits[i].topk(top_k).indices.tolist()]
            for i, tok in enumerate(ids) if tok == corpus.mask_id
        ]
