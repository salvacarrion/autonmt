"""SPM encode → translate → decode round-trip for SentencePiece-based backends.

Composition (not inheritance) over :class:`BaseTranslator`: backends that use
AutoNMT-style SentencePiece vocabularies (autonmt, fairseq) declare
``self._spm = SPMTranslatePipeline(...)`` in their constructor, and
``BaseTranslator.translate()`` delegates to it. Backends that own their own
tokenizer (huggingface) leave ``self._spm = None`` and write
``src.txt`` / ``ref.txt`` / ``hyp.txt`` directly from ``_translate``.

The pipeline owns:
  * the ``eval/<ds>/data/{0_raw, 1_preprocessed, 3_encoded}`` staging,
  * the per-(subset, beam) cache gate (``hyp.tok`` existence),
  * decoding ``hyp.tok`` back to ``hyp.txt``,
  * materializing ``src.txt`` / ``ref.txt`` from the raw test files
    (avoiding bias from model-emitted ``<unk>``),
  * the optional predict-time ``preprocess_fn`` round-trip,
  * the ref/hyp line-count assertion.

The backend's ``_translate`` hook only has to produce ``hyp.tok`` in
``output_path``.
"""
import datetime
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from autonmt.backends._base.run_layout import RunLayout
from autonmt.datasets.dataset import Dataset
from autonmt.datasets.encoding import decode_file, encode_file
from autonmt.datasets.preprocessing import preprocess_predict_file
from autonmt.utils.fileio import count_file_lines, make_dir, read_file_lines, write_file_lines
from autonmt.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-eval context
# ---------------------------------------------------------------------------

@dataclass
class SPMTranslateContext:
    """Per-eval snapshot consumed by every beam pass of one translate() call.

    Only SPM-based backends know about subword models / pretok flags / vocab
    files — hence this lives next to the pipeline rather than in
    :class:`BaseTranslator`.
    """
    checkpoints_dir: str
    model_eval_path: str
    dst_raw_path: str
    dst_preprocessed_path: str
    dst_encoded_path: str

    model_src_vocab_path: Optional[str]
    model_trg_vocab_path: Optional[str]

    vocab_langs: Tuple[str, str]
    pretok_flags: Dict[str, bool] = field(default_factory=dict)
    vocab_paths: Dict[str, str] = field(default_factory=dict)
    subword_models: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SPMTranslatePipeline:
    """SPM encode → backend ``_translate`` → decode round-trip."""

    def __init__(self, layout: RunLayout, src_vocab, trg_vocab,
                 test_subsets: List[Tuple[str, Optional[Callable]]]):
        self._layout = layout
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.test_subsets = test_subsets

    # --- Public entry point ---------------------------------------------

    def run(self, eval_ds: Dataset, beams: List[int],
            preprocess_fn: Optional[Callable],
            force_overwrite: bool,
            translate_fn: Callable[..., None],
            preprocess_eval_fn: Optional[Callable[..., None]] = None,
            **kwargs) -> None:
        """Orchestrate the full SPM translate flow for one ``eval_ds``.

        Parameters
        ----------
        eval_ds : Dataset
            Dataset to translate.
        beams : list of int
            Beam widths to run.
        preprocess_fn : callable or None
            User-supplied predict-time normalization (run before SPM encode
            and re-applied to src/ref/hyp text artifacts).
        force_overwrite : bool
            Cache gate for the (subset, beam) artifacts (``hyp.tok``).
        translate_fn : callable
            Backend hook. Called once per (subset, beam). Signature::

                translate_fn(*, data_path, output_path, src_lang, trg_lang,
                             beam_width, checkpoints_dir,
                             model_src_vocab_path, model_trg_vocab_path,
                             filter_idx, **kwargs) -> None

            Must write ``hyp.tok`` into ``output_path``. The pipeline decodes
            it into ``hyp.txt`` and materializes ``src.txt`` / ``ref.txt``.
        preprocess_eval_fn : callable or None
            Optional pre-translate hook for backends that need extra eval
            artifacts (e.g. fairseq's data-bin, autonmt's test
            ``TranslationDataset``). Signature mirrors the historical
            ``_preprocess(..., apply2test=True)`` call.
        """
        ctx = self._build_context(eval_ds)
        self._encode_eval_text(eval_ds, ctx, preprocess_fn=preprocess_fn,
                                force_overwrite=force_overwrite)
        if preprocess_eval_fn is not None:
            self._invoke_preprocess_eval(eval_ds, ctx, preprocess_eval_fn,
                                          force_overwrite=force_overwrite, **kwargs)

        for filter_idx, (fn_name, filter_fn) in enumerate(self.test_subsets):
            for beam in beams:
                self._translate_one_beam(
                    eval_ds=eval_ds, ctx=ctx, beam=beam,
                    filter_idx=filter_idx, filter_fn=filter_fn, fn_name=fn_name,
                    preprocess_fn=preprocess_fn,
                    translate_fn=translate_fn,
                    force_overwrite=force_overwrite, **kwargs,
                )

    # --- Build context --------------------------------------------------

    def _build_context(self, eval_ds) -> SPMTranslateContext:
        model_eval_path = self._layout.eval_path(str(eval_ds))
        make_dir([model_eval_path])

        dst_raw_path = self._layout.eval_raw_path(str(eval_ds))
        dst_preprocessed_path = self._layout.eval_preprocessed_path(str(eval_ds))
        dst_encoded_path = self._layout.eval_encoded_path(str(eval_ds))
        make_dir([dst_raw_path, dst_preprocessed_path, dst_encoded_path])

        src_lang, trg_lang = self.src_vocab.lang, self.trg_vocab.lang
        return SPMTranslateContext(
            checkpoints_dir=self._layout.checkpoints_path(),
            model_eval_path=model_eval_path,
            dst_raw_path=dst_raw_path,
            dst_preprocessed_path=dst_preprocessed_path,
            dst_encoded_path=dst_encoded_path,
            model_src_vocab_path=self.src_vocab.vocab_path,
            model_trg_vocab_path=self.trg_vocab.vocab_path,
            vocab_langs=(src_lang, trg_lang),
            pretok_flags={src_lang: self.src_vocab.pretok_flag,
                          trg_lang: self.trg_vocab.pretok_flag},
            vocab_paths={src_lang: self.src_vocab.model_path,
                         trg_lang: self.trg_vocab.model_path},
            subword_models={src_lang: self.src_vocab.subword_model,
                            trg_lang: self.trg_vocab.subword_model},
        )

    # --- Encode eval text (copy raw → preprocess → SPM encode) ----------

    def _encode_eval_text(self, eval_ds, ctx: SPMTranslateContext,
                          preprocess_fn, force_overwrite):
        test_fnames = [f"{eval_ds.test_name}.{eval_ds.src_lang}",
                       f"{eval_ds.test_name}.{eval_ds.trg_lang}"]
        for i, ts_fname in enumerate(test_fnames):
            input_file = eval_ds.get_split_path(ts_fname)
            input_lang = ts_fname.split(".")[-1]
            vocab_lang = ctx.vocab_langs[i]

            # 1 - copy raw
            source_file = os.path.join(ctx.dst_raw_path, ts_fname)
            if force_overwrite or not os.path.exists(source_file):
                shutil.copyfile(input_file, source_file)
                assert os.path.exists(source_file)
                input_file = source_file

            # 2 - preprocess (+ pretokenize)
            preprocessed_file = os.path.join(ctx.dst_preprocessed_path, ts_fname)
            preprocess_predict_file(
                input_file=input_file, output_file=preprocessed_file,
                preprocess_fn=preprocess_fn,
                pretokenize=ctx.pretok_flags[vocab_lang],
                input_lang=input_lang, vocab_lang=vocab_lang,
                ds=eval_ds, force_overwrite=force_overwrite,
            )

            # 3 - subword encode
            enc_file = os.path.join(ctx.dst_encoded_path, ts_fname)
            encode_file(
                input_file=preprocessed_file, output_file=enc_file,
                model_vocab_path=ctx.vocab_paths[vocab_lang],
                subword_model=ctx.subword_models[vocab_lang],
                force_overwrite=force_overwrite,
            )

    # --- Pre-translate hook for backends needing extra eval artifacts ---

    def _invoke_preprocess_eval(self, eval_ds, ctx: SPMTranslateContext,
                                  preprocess_eval_fn,
                                  force_overwrite, **kwargs):
        test_path = os.path.join(ctx.dst_encoded_path, eval_ds.test_name)
        # ``ds=eval_ds`` is passed in addition to the historical signature:
        # autonmt's hook swallows it via **kwargs, fairseq's hook needs it to
        # build the data-bin path. Closes a latent bug where the legacy base
        # call dropped ``ds`` and fairseq's eval-time _preprocess crashed.
        preprocess_eval_fn(
            ds=eval_ds,
            train_path=None, val_path=None, test_path=test_path,
            src_lang=eval_ds.src_lang, trg_lang=eval_ds.trg_lang,
            src_vocab_path=ctx.model_src_vocab_path,
            trg_vocab_path=ctx.model_trg_vocab_path,
            apply2train=False, apply2val=False, apply2test=True,
            output_path=ctx.model_eval_path,
            force_overwrite=force_overwrite, **kwargs,
        )

    # --- Per-(subset, beam) loop ----------------------------------------

    def _translate_one_beam(self, eval_ds, ctx: SPMTranslateContext, beam,
                             filter_idx, filter_fn, fn_name,
                             preprocess_fn, translate_fn,
                             force_overwrite, **kwargs):
        extra_str = f" | split='{fn_name}'" if fn_name else ""
        output_path = self._layout.beam_path(
            str(eval_ds), fn_name, beam)
        make_dir(output_path)

        if not force_overwrite and os.path.exists(os.path.join(output_path, "hyp.tok")):
            return

        start = time.time()
        # NOTE: ``force_overwrite`` is *not* forwarded to ``translate_fn`` —
        # the cache gate above (hyp.tok existence) is the single source of
        # truth. The backend's job is just to produce the artifacts.
        # ``ds=eval_ds`` lets fairseq compute the dataset-side data-bin
        # path; autonmt swallows it via **kwargs.
        translate_fn(
            ds=eval_ds,
            data_path=ctx.model_eval_path, output_path=output_path,
            src_lang=eval_ds.src_lang, trg_lang=eval_ds.trg_lang,
            beam_width=beam, checkpoints_dir=ctx.checkpoints_dir,
            model_src_vocab_path=ctx.model_src_vocab_path,
            model_trg_vocab_path=ctx.model_trg_vocab_path,
            filter_idx=filter_idx, **kwargs,
        )

        src_output_file = os.path.join(output_path, "src.txt")
        ref_output_file = os.path.join(output_path, "ref.txt")
        hyp_output_file = os.path.join(output_path, "hyp.txt")

        self._decode_hypothesis(ctx, output_path, hyp_output_file, force_overwrite)
        self._materialize_src_ref(eval_ds, ctx, src_output_file, ref_output_file,
                                   filter_fn=filter_fn, fn_name=fn_name)
        if preprocess_fn:
            self._postprocess_eval_files(eval_ds, ctx, preprocess_fn,
                                          src_output_file, ref_output_file, hyp_output_file)
        self._assert_ref_hyp_line_count(output_path)

        log.info(f"\t- Translating time (beam={beam}{extra_str}): "
                 f"{datetime.timedelta(seconds=time.time() - start)}")

    def _decode_hypothesis(self, ctx: SPMTranslateContext, output_path,
                            hyp_output_file, force_overwrite):
        model_lang = self.trg_vocab.lang
        hyp_input_file = os.path.join(output_path, "hyp.tok")
        decode_file(
            input_file=hyp_input_file, output_file=hyp_output_file, lang=model_lang,
            subword_model=ctx.subword_models[model_lang],
            pretok_flag=ctx.pretok_flags[model_lang],
            model_vocab_path=ctx.vocab_paths[model_lang],
            remove_unk_hyphen=True, force_overwrite=force_overwrite,
        )

    def _materialize_src_ref(self, eval_ds, ctx: SPMTranslateContext,
                              src_output_file, ref_output_file,
                              filter_fn, fn_name):
        src_input_file = os.path.join(ctx.dst_raw_path,
                                       f"{eval_ds.test_name}.{eval_ds.src_lang}")
        ref_input_file = os.path.join(ctx.dst_raw_path,
                                       f"{eval_ds.test_name}.{eval_ds.trg_lang}")
        if not filter_fn:
            shutil.copyfile(src_input_file, src_output_file)
            shutil.copyfile(ref_input_file, ref_output_file)
            return

        log.info(f"Filtering src/ref raw files (split='{fn_name}')...")
        src_lines = read_file_lines(filename=src_input_file, autoclean=True)
        trg_lines = read_file_lines(filename=ref_input_file, autoclean=True)
        src_lines, trg_lines = filter_fn(src_lines, trg_lines, from_fn="translate")
        write_file_lines(filename=src_output_file, lines=src_lines,
                         autoclean=True, insert_break_line=True)
        write_file_lines(filename=ref_output_file, lines=trg_lines,
                         autoclean=True, insert_break_line=True)

    def _postprocess_eval_files(self, eval_ds, ctx: SPMTranslateContext,
                                 preprocess_fn,
                                 src_output_file, ref_output_file, hyp_output_file):
        # force_overwrite must be True here to rewrite src/ref/hyp in-place.
        for path, vocab_lang, lang in (
            (src_output_file, self.src_vocab.lang, eval_ds.src_lang),
            (ref_output_file, self.trg_vocab.lang, eval_ds.trg_lang),
            (hyp_output_file, self.trg_vocab.lang, eval_ds.trg_lang),
        ):
            preprocess_predict_file(
                input_file=path, output_file=path,
                preprocess_fn=preprocess_fn,
                pretokenize=ctx.pretok_flags[vocab_lang],
                input_lang=lang, vocab_lang=vocab_lang,
                ds=eval_ds, force_overwrite=True,
            )

    @staticmethod
    def _assert_ref_hyp_line_count(output_path):
        n_ref = count_file_lines(os.path.join(output_path, "ref.txt"))
        n_hyp = count_file_lines(os.path.join(output_path, "hyp.txt"))
        if n_ref != n_hyp:
            raise ValueError(
                f"The number of lines in 'ref.txt' ({n_ref}) and 'hyp.txt' "
                f"({n_hyp}) does not match. If you see a 'CUDA out of memory' "
                f"message, try again with smaller batch.")


__all__ = ["SPMTranslatePipeline", "SPMTranslateContext"]
