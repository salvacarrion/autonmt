"""Fairseq backend was deprecated when Meta archived the repo (2026-03-20).

The deprecation must be visible at two points:
  - importing the module emits a DeprecationWarning
  - instantiating FairseqTranslator without `fairseq` installed raises an
    informative ImportError (not the silent NameError it used to)
"""
import importlib
import sys
import warnings

import pytest


def _reload_fairseq_module():
    """Force re-import so the module-level DeprecationWarning fires under catch_warnings."""
    sys.modules.pop("autonmt.toolkits.fairseq", None)
    return importlib.import_module("autonmt.toolkits.fairseq")


class TestModuleLevelDeprecation:
    def test_importing_module_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _reload_fairseq_module()
        msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("FairseqTranslator is deprecated" in m for m in msgs), \
            f"expected DeprecationWarning, got: {msgs}"

    def test_warning_mentions_archived_date(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _reload_fairseq_module()
        msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("2026-03-20" in m for m in msgs)


@pytest.mark.skipif(
    importlib.util.find_spec("fairseq") is not None,
    reason="fairseq is installed; this test asserts the missing-dep error path",
)
class TestInstantiationWithoutFairseq:
    def test_init_raises_importerror(self):
        mod = _reload_fairseq_module()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ImportError, match="Fairseq is not installed"):
                mod.FairseqTranslator(runs_dir="/tmp/x", run_name="x")

    def test_error_message_includes_install_hint(self):
        mod = _reload_fairseq_module()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ImportError, match="pip install fairseq"):
                mod.FairseqTranslator(runs_dir="/tmp/x", run_name="x")
