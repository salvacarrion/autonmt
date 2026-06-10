"""Single-stream / instruct corpora for language modelling.

Sibling of :class:`~autonmt.datasets.parallel.corpus.ParallelCorpus` (which is parallel-text,
``xx-yy``-keyed). This deliberately does **not** touch that class: an LM corpus
has one stream, not a language pair, so it gets its own identity and on-disk
layout. It *reuses* the v1.0 SentencePiece machinery
(:func:`autonmt.datasets.processing.tokenizers.spm_train_file` and the ``sentencepiece``
runtime) — only the ingest/split/pack stages are LM-specific.

Three modes:
  * ``"text"``   — one document per line; trains a (decoder-only) LM on the raw stream.
  * ``"instruct"`` — parallel ``prompt`` / ``completion`` lines; the prompt span
    is masked out of the loss so the model only learns the completion.
  * ``"mlm"``    — like ``"text"`` (single stream), but reserves a ``<mask>`` token in the
    vocabulary for masked language modelling (BERT-style encoder-only training).

On-disk layout (under ``base_path/<name>/<size>/``)::

    data/0_raw/        text: data.txt           | instruct: data.prompt + data.completion
    data/1_splits/     text: {train,val}.txt    | instruct: {train,val}.{prompt,completion}
    data/4_encoded/<sw>/<vs>/  {train,val}.tokens.npy   (+ .sup.npy for instruct)
    vocabs/<sw>/<vs>/  spm.model + spm.vocab
    models/<engine>/runs/<run>/...   checkpoints, logs (engine = backend's ENGINE, e.g. autonmt)

This module is just the identity/layout holder; :class:`LMCorpus` describes
where each stage lives on disk. The flat token-id streams are produced by
:class:`~autonmt.datasets.lm.builder.LMCorpusBuilder` and consumed at read-time
by :class:`~autonmt.core.data.lm_dataset.LMDataset`, which packs them into fixed
blocks.

References
----------
Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.*
(packed next-token pretraining on a concatenated token stream)
[OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Ouyang et al. (2022). *Training Language Models to Follow Instructions with
Human Feedback.* (instruct prompt→completion supervision, i.e. the masked-prompt
mode) [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
"""
import os

from autonmt.utils.enums import SubwordModel


# SentencePiece is trained with pad_id=3 (see tokenizers.spm_train_file); its
# defaults put unk=0, bos=1, eos=2. We mirror those so the ids written into the
# packed stream agree with what the model's embedding table expects.
UNK_ID, SOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3

TEXT, INSTRUCT, MLM = "text", "instruct", "mlm"


class LMCorpus:
    """Identity + on-disk paths for one LM corpus variant (name × size × subword × vocab)."""

    def __init__(self, base_path, name, size_name, mode, subword_model, vocab_size,
                 byte_fallback=False, train_name="train", val_name="val"):
        if mode not in (TEXT, INSTRUCT, MLM):
            raise ValueError(f"Unknown LM corpus mode {mode!r} (expected {TEXT!r}, {INSTRUCT!r} or {MLM!r})")
        sw, sugar_bf = SubwordModel.parse_with_byte_fallback(subword_model, default_byte_fallback=byte_fallback)
        if sw is None or not sw.uses_sentencepiece:
            raise ValueError(
                f"LM corpora require a SentencePiece subword model "
                f"(word/bpe/unigram/char), got {subword_model!r}."
            )

        self.base_path = base_path
        self.name = name.strip()
        self.size_name = size_name.strip()
        self.mode = mode
        self.subword_model = sw
        self.byte_fallback = bool(sugar_bf)
        self.vocab_size = str(vocab_size).lower()
        self.train_name = train_name
        self.val_name = val_name

        # Special-token ids (mirrors the trained SentencePiece model).
        self.unk_id, self.sos_id, self.eos_id, self.pad_id = UNK_ID, SOS_ID, EOS_ID, PAD_ID
        # Reserved whole-piece mask token for MLM mode (trained into the vocab via
        # user_defined_symbols); its id is looked up from the model — see ``mask_id``.
        self.mask_piece = "<mask>"

        self._sp = None  # lazily loaded SentencePiece processor

    # --- Identity --------------------------------------------------------

    def __str__(self):
        return "_".join([self.name, self.size_name, self._sw_str(), self.vocab_size]).lower()

    def _sw_str(self):
        return f"{self.subword_model}+bytes" if self.byte_fallback else str(self.subword_model)

    def _vocab_size_id(self):
        return (self._sw_str(), self.vocab_size)

    # --- Paths -----------------------------------------------------------

    def _root(self, *parts):
        return os.path.join(self.base_path, self.name, self.size_name, *parts)

    def get_raw_path(self, fname=""):
        return self._root("data", "0_raw", fname)

    def get_splits_path(self, fname=""):
        return self._root("data", "1_splits", fname)

    def get_encoded_path(self, fname=""):
        return self._root("data", "4_encoded", *self._vocab_size_id(), fname)

    def get_vocab_path(self, fname=""):
        return self._root("vocabs", *self._vocab_size_id(), fname)

    def get_runs_path(self, engine, fname=""):
        # ``engine`` is supplied by the backend (its ENGINE constant), mirroring
        # ParallelCorpus.get_runs_path(toolkit=...) — so models/<engine>/ tracks the
        # backend that wrote the run (autonmt, huggingface), not the data.
        return self._root("models", engine, "runs", fname)

    def get_run_name(self, run_prefix):
        return f"{run_prefix}_{self._sw_str()}_{self.vocab_size}".lower()

    # --- Stage filenames -------------------------------------------------

    def raw_files(self):
        """Raw filenames: one stream for text/mlm, two (prompt/completion) for instruct."""
        if self.mode != INSTRUCT:
            return ["data.txt"]
        return ["data.prompt", "data.completion"]

    def split_files(self, split):
        if self.mode != INSTRUCT:
            return [f"{split}.txt"]
        return [f"{split}.prompt", f"{split}.completion"]

    def spm_prefix(self):
        return self.get_vocab_path("spm")

    def spm_model_path(self):
        return self.spm_prefix() + ".model"

    def tokens_file(self, split):
        return self.get_encoded_path(f"{split}.tokens.npy")

    def supervise_file(self, split):
        return self.get_encoded_path(f"{split}.sup.npy")

    @property
    def splits(self):
        return (self.train_name, self.val_name)

    # --- SentencePiece runtime (used by the trainer for prompts) ---------

    def load_spm(self):
        if self._sp is None:
            import sentencepiece as spm
            self._sp = spm.SentencePieceProcessor(model_file=self.spm_model_path())
        return self._sp

    @property
    def model_vocab_size(self):
        """Number of pieces in the trained model — the model's vocab dimension."""
        return self.load_spm().get_piece_size()

    @property
    def mask_id(self):
        """Id of the reserved ``<mask>`` piece (MLM mode only); ``None`` otherwise."""
        if self.mode != MLM:
            return None
        return self.load_spm().piece_to_id(self.mask_piece)

    def special_ids(self):
        """Special-token ids to exclude from MLM masking candidates."""
        ids = {self.unk_id, self.sos_id, self.eos_id, self.pad_id}
        if self.mode == MLM:
            ids.add(self.mask_id)
        return ids

    def encode(self, text, add_sos=True, add_eos=False):
        ids = self.load_spm().encode(text, out_type=int)
        if add_sos:
            ids = [self.sos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        # Drop special tokens before SentencePiece detokenisation.
        specials = {self.unk_id, self.sos_id, self.eos_id, self.pad_id}
        ids = [int(i) for i in ids if int(i) not in specials]
        return self.load_spm().decode(ids)
