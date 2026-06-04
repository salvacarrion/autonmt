"""Single-stream / instruct corpora for language modelling.

Sibling of :class:`~autonmt.datasets.dataset.Dataset` (which is parallel-text,
``xx-yy``-keyed). This deliberately does **not** touch that class: an LM corpus
has one stream, not a language pair, so it gets its own identity and on-disk
layout. It *reuses* the v1.0 SentencePiece machinery
(:func:`autonmt.datasets.tokenizers.spm_train_file` and the ``sentencepiece``
runtime) — only the ingest/split/pack stages are LM-specific.

Two modes:
  * ``"text"``   — one document per line; trains an LM on the raw stream.
  * ``"instruct"`` — parallel ``prompt`` / ``completion`` lines; the prompt span
    is masked out of the loss so the model only learns the completion.

On-disk layout (under ``base_path/<name>/<size>/``)::

    data/0_raw/        text: data.txt           | instruct: data.prompt + data.completion
    data/1_splits/     text: {train,val}.txt    | instruct: {train,val}.{prompt,completion}
    data/4_encoded/<sw>/<vs>/  {train,val}.tokens.npy   (+ .sup.npy for instruct)
    vocabs/<sw>/<vs>/  spm.model + spm.vocab
    models/lm/runs/<run>/...   checkpoints, logs (written by LMTrainer)

Packing into fixed blocks is a read-time concern handled by
:class:`~autonmt.core.data.lm_dataset.LMDataset`; this module produces the flat
token-id streams it consumes.

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

import numpy as np

from autonmt.datasets import tokenizers
from autonmt.utils.enums import SubwordModel
from autonmt.utils.fileio import make_dir, read_file_lines, write_file_lines
from autonmt.utils.logger import get_logger

log = get_logger(__name__)


# SentencePiece is trained with pad_id=3 (see tokenizers.spm_train_file); its
# defaults put unk=0, bos=1, eos=2. We mirror those so the ids written into the
# packed stream agree with what the model's embedding table expects.
UNK_ID, SOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3

TEXT, INSTRUCT = "text", "instruct"


class LMCorpus:
    """Identity + on-disk paths for one LM corpus variant (name × size × subword × vocab)."""

    def __init__(self, base_path, name, size_name, mode, subword_model, vocab_size,
                 byte_fallback=False, train_name="train", val_name="val",
                 engine="lm"):
        if mode not in (TEXT, INSTRUCT):
            raise ValueError(f"Unknown LM corpus mode {mode!r} (expected {TEXT!r} or {INSTRUCT!r})")
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
        self.engine = engine

        # Special-token ids (mirrors the trained SentencePiece model).
        self.unk_id, self.sos_id, self.eos_id, self.pad_id = UNK_ID, SOS_ID, EOS_ID, PAD_ID

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

    def get_runs_path(self, fname=""):
        return self._root("models", self.engine, "runs", fname)

    def get_run_name(self, run_prefix):
        return f"{run_prefix}_{self._sw_str()}_{self.vocab_size}".lower()

    # --- Stage filenames -------------------------------------------------

    def raw_files(self):
        """Raw filenames for this corpus mode: one for text, two for instruct."""
        if self.mode == TEXT:
            return ["data.txt"]
        return ["data.prompt", "data.completion"]

    def split_files(self, split):
        if self.mode == TEXT:
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


class LMCorpusBuilder:
    """Unrolls (corpus × subword × vocab) and materialises packed token streams.

    Mirrors :class:`~autonmt.datasets.dataset_builder.DatasetBuilder` for the LM
    case. Each corpus declaration is a dict::

        {"name": "tiny", "mode": "text", "sizes": [("original", None)],
         "text": ["line 1", "line 2", ...]}          # inline source (optional)

        {"name": "qa", "mode": "instruct", "sizes": [("original", None)],
         "pairs": [("prompt", "completion"), ...]}    # inline source (optional)

    ``text`` / ``pairs`` are convenience inline sources written to ``0_raw`` when
    no raw files exist yet (handy for examples/tests); otherwise place the raw
    files on disk under ``data/0_raw/`` following :meth:`LMCorpus.raw_files`.

    Parameters
    ----------
    base_path : str
        Root directory for all corpus variants.
    corpus : list of dict
        Corpus declarations (see above).
    encoding : list of dict
        ``{"subword_models": [...], "vocab_sizes": [...]}`` axes, as in
        :class:`DatasetBuilder`.
    val_size : int or float, default 0.1
        Validation split. ``< 1`` is treated as a fraction of the corpus,
        ``>= 1`` as an absolute number of lines.
    character_coverage, input_sentence_size, split_digits :
        SentencePiece training knobs (same meaning as in ``DatasetBuilder``).
    """

    INPUT_SENTENCE_SIZE = 1_000_000
    CHARACTER_COVERAGE = 1.0
    SPLIT_DIGITS = True

    def __init__(self, base_path, corpus, encoding, val_size=0.1,
                 character_coverage=None, input_sentence_size=None, split_digits=None):
        self.base_path = base_path
        self.corpus = corpus
        self.encoding = encoding
        self.val_size = val_size
        self.character_coverage = character_coverage if character_coverage is not None else self.CHARACTER_COVERAGE
        self.input_sentence_size = input_sentence_size if input_sentence_size is not None else self.INPUT_SENTENCE_SIZE
        self.split_digits = split_digits if split_digits is not None else self.SPLIT_DIGITS

        self.corpora = self._unroll()

    # --- Cross-product unrolling ----------------------------------------

    def _unroll(self):
        encs = self._unroll_encoding(self.encoding)
        out = []
        for c in self.corpus:
            mode = c.get("mode", TEXT)
            for size_name, _ in c.get("sizes", [("original", None)]):
                for enc in encs:
                    out.append(LMCorpus(
                        base_path=self.base_path, name=c["name"], size_name=size_name,
                        mode=mode, subword_model=enc["subword_model"],
                        vocab_size=enc["vocab_size"], byte_fallback=enc["byte_fallback"],
                    ))
        return out

    @staticmethod
    def _unroll_encoding(encoding):
        encs, seen = [], set()
        for entry in encoding:
            entry_bf = bool(entry.get("byte_fallback", False))
            for model_raw in entry["subword_models"]:
                model, bf = SubwordModel.parse_with_byte_fallback(model_raw, default_byte_fallback=entry_bf)
                for size in entry["vocab_sizes"]:
                    key = (str(model), size, bf)
                    if key in seen:
                        continue
                    seen.add(key)
                    encs.append({"subword_model": str(model), "vocab_size": size, "byte_fallback": bf})
        return encs

    # --- Accessors ------------------------------------------------------

    def __iter__(self):
        return iter(self.corpora)

    def __len__(self):
        return len(self.corpora)

    def get_train_ds(self):
        return self.corpora

    # --- Build ----------------------------------------------------------

    def build(self, force_overwrite=False):
        log.info(f"=> Building LM corpora... (base_path={self.base_path})")
        # Raw ingest + split are subword-independent, so do them once per
        # (name, size) instead of once per encoding cell.
        done_splits = set()
        for c in self.corpora:
            key = (c.name, c.size_name, c.mode)
            if key not in done_splits:
                self._write_inline_raw(c, force_overwrite=force_overwrite)
                self._create_splits(c, force_overwrite=force_overwrite)
                done_splits.add(key)
            self._train_spm(c, force_overwrite=force_overwrite)
            self._encode_and_pack(c, force_overwrite=force_overwrite)
        return self

    def _decl_for(self, corpus):
        for c in self.corpus:
            if c["name"] == corpus.name:
                return c
        return {}

    def _write_inline_raw(self, corpus, force_overwrite):
        """Materialise inline ``text`` / ``pairs`` into 0_raw (examples/tests)."""
        decl = self._decl_for(corpus)
        raw_paths = [corpus.get_raw_path(f) for f in corpus.raw_files()]
        if not force_overwrite and all(os.path.exists(p) for p in raw_paths):
            return
        make_dir(corpus.get_raw_path())

        if corpus.mode == TEXT:
            lines = decl.get("text")
            if lines is None:
                if not os.path.exists(raw_paths[0]):
                    raise FileNotFoundError(
                        f"No inline 'text' and no raw file at {raw_paths[0]!r}. "
                        f"Provide one to build corpus {corpus.name!r}."
                    )
                return
            write_file_lines(lines, filename=raw_paths[0], insert_break_line=True)
        else:
            pairs = decl.get("pairs")
            if pairs is None:
                if not all(os.path.exists(p) for p in raw_paths):
                    raise FileNotFoundError(
                        f"No inline 'pairs' and missing raw files {raw_paths}. "
                        f"Provide them to build corpus {corpus.name!r}."
                    )
                return
            prompts = [p for p, _ in pairs]
            completions = [c for _, c in pairs]
            write_file_lines(prompts, filename=raw_paths[0], insert_break_line=True)
            write_file_lines(completions, filename=raw_paths[1], insert_break_line=True)

    def _resolve_val_lines(self, n_total):
        v = self.val_size
        n_val = int(round(n_total * v)) if v < 1 else int(v)
        n_val = max(0, min(n_val, n_total - 1))  # always leave at least 1 train line
        return n_val

    def _create_splits(self, corpus, force_overwrite):
        out_paths = [corpus.get_splits_path(f)
                     for split in corpus.splits for f in corpus.split_files(split)]
        if not force_overwrite and all(os.path.exists(p) for p in out_paths):
            return
        make_dir(corpus.get_splits_path())

        # Read each raw "column" (1 file for text, 2 for instruct) and split the
        # last n_val lines into val. All columns share the same split boundary.
        raw_cols = [read_file_lines(corpus.get_raw_path(f), autoclean=True)
                    for f in corpus.raw_files()]
        n_total = len(raw_cols[0])
        for col in raw_cols:
            if len(col) != n_total:
                raise ValueError(
                    f"Raw files for corpus {corpus.name!r} have mismatched line counts "
                    f"({[len(c) for c in raw_cols]}); prompt/completion must be aligned."
                )
        n_val = self._resolve_val_lines(n_total)

        for col, raw_fname in zip(raw_cols, corpus.raw_files()):
            train_lines = col[:n_total - n_val] if n_val else col
            val_lines = col[n_total - n_val:] if n_val else []
            suffix = raw_fname.split(".", 1)[1]  # "txt" | "prompt" | "completion"
            write_file_lines(train_lines, corpus.get_splits_path(f"{corpus.train_name}.{suffix}"),
                             insert_break_line=True)
            write_file_lines(val_lines, corpus.get_splits_path(f"{corpus.val_name}.{suffix}"),
                             insert_break_line=True)
        log.info(f"\t- Split corpus {corpus.name!r}: train={n_total - n_val}, val={n_val}")

    def _spm_train_input(self, corpus):
        """Text file SentencePiece is trained on (completion+prompt concatenated for instruct)."""
        if corpus.mode == TEXT:
            return corpus.get_splits_path(f"{corpus.train_name}.txt")
        # Instruct: train the tokenizer on both sides so prompt vocabulary is covered.
        merged = corpus.get_vocab_path("_spm_train_input.txt")
        make_dir(corpus.get_vocab_path())
        lines = (read_file_lines(corpus.get_splits_path(f"{corpus.train_name}.prompt"), autoclean=True)
                 + read_file_lines(corpus.get_splits_path(f"{corpus.train_name}.completion"), autoclean=True))
        write_file_lines(lines, merged, insert_break_line=True)
        return merged

    def _train_spm(self, corpus, force_overwrite):
        if not force_overwrite and os.path.exists(corpus.spm_model_path()):
            return
        make_dir(corpus.get_vocab_path())
        log.info(f"\t- Training SentencePiece: {corpus} (subword={corpus._sw_str()}, vocab={corpus.vocab_size})")
        tokenizers.spm_train_file(
            input_file=self._spm_train_input(corpus),
            model_prefix=corpus.spm_prefix(),
            subword_model=corpus.subword_model,
            vocab_size=corpus.vocab_size,
            input_sentence_size=self.input_sentence_size,
            character_coverage=self.character_coverage,
            split_digits=self.split_digits,
            byte_fallback=corpus.byte_fallback,
        )
        assert os.path.exists(corpus.spm_model_path())

    def _encode_and_pack(self, corpus, force_overwrite):
        make_dir(corpus.get_encoded_path())
        sp = corpus.load_spm()
        # uint16 covers vocab < 65536 (the common case); fall back to uint32.
        dtype = np.uint16 if corpus.model_vocab_size < (1 << 16) else np.uint32

        for split in corpus.splits:
            tokens_path = corpus.tokens_file(split)
            if not force_overwrite and os.path.exists(tokens_path):
                continue
            if corpus.mode == TEXT:
                self._pack_text(corpus, sp, split, tokens_path, dtype)
            else:
                self._pack_instruct(corpus, sp, split, tokens_path, dtype)

    def _pack_text(self, corpus, sp, split, tokens_path, dtype):
        lines = read_file_lines(corpus.get_splits_path(f"{split}.txt"), autoclean=True)
        stream = []
        for line in lines:
            # [<s>] doc [</s>] per document so the model sees clean boundaries
            # and generation can seed from <s>.
            stream.append(corpus.sos_id)
            stream.extend(sp.encode(line, out_type=int))
            stream.append(corpus.eos_id)
        self._save_stream(tokens_path, stream, dtype)
        log.info(f"\t- Packed {split!r} ({len(lines)} docs → {len(stream)} tokens): {tokens_path}")

    def _pack_instruct(self, corpus, sp, split, tokens_path, dtype):
        prompts = read_file_lines(corpus.get_splits_path(f"{split}.prompt"), autoclean=True)
        completions = read_file_lines(corpus.get_splits_path(f"{split}.completion"), autoclean=True)
        assert len(prompts) == len(completions)
        stream, supervise = [], []
        for prompt, completion in zip(prompts, completions):
            p_ids = [corpus.sos_id] + sp.encode(prompt, out_type=int)
            c_ids = sp.encode(completion, out_type=int) + [corpus.eos_id]
            stream.extend(p_ids + c_ids)
            # Mask the prompt span: only the completion (+ </s>) contributes to loss.
            supervise.extend([0] * len(p_ids) + [1] * len(c_ids))
        self._save_stream(tokens_path, stream, dtype)
        self._save_stream(corpus.supervise_file(split), supervise, np.uint8)
        log.info(f"\t- Packed instruct {split!r} ({len(prompts)} pairs → {len(stream)} tokens): {tokens_path}")

    @staticmethod
    def _save_stream(path, values, dtype):
        np.save(path, np.asarray(values, dtype=dtype))
        # np.save appends .npy; normalise so callers reading the exact path work.
        if not os.path.exists(path) and os.path.exists(path + ".npy"):
            os.replace(path + ".npy", path)
