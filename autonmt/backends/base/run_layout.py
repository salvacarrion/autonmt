"""Path engine for one training run.

Mirrors :class:`~autonmt.datasets.dataset.DatasetLayout` for the run-side
of the on-disk layout::

    runs_dir/<run_name>/checkpoints/
    runs_dir/<run_name>/logs/
    runs_dir/<run_name>/eval/<eval_name>/
                              data/0_raw/
                              data/1_preprocessed/
                              data/3_encoded/
                              translations/beam<N>/
                              translations/beam<N>/scores/

Pure path computation — no I/O. Lets every translator and report consumer agree
on where files live without each one re-implementing the layout.
"""
import os
from typing import Optional


class RunLayout:

    CHECKPOINTS = "checkpoints"
    LOGS = "logs"
    EVAL = "eval"
    TRANSLATIONS = "translations"
    BEAM = "beam"
    SCORES = "scores"

    # eval/.../data sub-tree
    EVAL_DATA = "data"
    EVAL_RAW = "0_raw"
    EVAL_PREPROCESSED = "1_preprocessed"
    EVAL_ENCODED = "3_encoded"

    def __init__(self, runs_dir: str, run_name: str):
        self.runs_dir = runs_dir
        self.run_name = run_name

    def _run(self, *parts: str) -> str:
        return os.path.join(self.runs_dir, self.run_name, *parts)

    # --- Per-run paths --------------------------------------------------

    def checkpoints_path(self, fname: str = "") -> str:
        return self._run(self.CHECKPOINTS, fname)

    def logs_path(self, fname: str = "") -> str:
        return self._run(self.LOGS, fname)

    def eval_path(self, eval_name: str, fname: str = "") -> str:
        return self._run(self.EVAL, eval_name, fname)

    def eval_data_bin_path(self, eval_name: str, data_bin_name: str, fname: str = "") -> str:
        return os.path.join(self.eval_path(eval_name), data_bin_name, fname)

    # --- Per-eval-dataset translation sub-tree --------------------------

    def translations_path(self, eval_name: str, split_name: Optional[str] = "") -> str:
        return os.path.join(self.eval_path(eval_name), self.TRANSLATIONS, split_name or "")

    def beam_path(self, eval_name: str, split_name: Optional[str], beam: int,
                  fname: str = "") -> str:
        return os.path.join(self.translations_path(eval_name, split_name),
                            self.BEAM, f"beam{beam}", fname)

    def beam_scores_path(self, eval_name: str, split_name: Optional[str], beam: int,
                         fname: str = "") -> str:
        return os.path.join(self.beam_path(eval_name, split_name, beam),
                            self.SCORES, fname)

    # --- Per-eval-dataset raw/preprocessed/encoded sub-tree -------------

    def eval_raw_path(self, eval_name: str) -> str:
        return os.path.join(self.eval_path(eval_name), self.EVAL_DATA, self.EVAL_RAW)

    def eval_preprocessed_path(self, eval_name: str) -> str:
        return os.path.join(self.eval_path(eval_name), self.EVAL_DATA, self.EVAL_PREPROCESSED)

    def eval_encoded_path(self, eval_name: str) -> str:
        return os.path.join(self.eval_path(eval_name), self.EVAL_DATA, self.EVAL_ENCODED)
