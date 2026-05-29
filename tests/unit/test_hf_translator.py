"""Phase 1 HuggingFaceTranslator: inference-only contract.

These tests verify the contract without actually loading a model from the Hub:
  - import the module without `transformers` raises a clear ImportError
  - the backend exposes the expected ENGINE id + lazy lookup via `autonmt.backends`
  - `from_dataset` auto-fills src_lang / tgt_lang from the dataset
  - `fit()` raises NotImplementedError (fine-tuning is Phase 2)
"""
import importlib
import sys
from types import SimpleNamespace

import pytest


def _reload_hf_module():
    sys.modules.pop("autonmt.backends.huggingface.translation_engine", None)
    sys.modules.pop("autonmt.backends.huggingface", None)
    return importlib.import_module("autonmt.backends.huggingface.translation_engine")


def test_lazy_export_from_backends():
    """`from autonmt.backends import HuggingFaceTranslator` works without instantiating."""
    import autonmt.backends as backends
    assert hasattr(backends, "HuggingFaceTranslator")
    cls = backends.HuggingFaceTranslator
    assert cls.__name__ == "HuggingFaceTranslator"
    assert cls.ENGINE == "huggingface"


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is not None,
    reason="transformers is installed; this test asserts the missing-dep error path",
)
class TestInstantiationWithoutTransformers:
    def test_init_raises_importerror(self):
        mod = _reload_hf_module()
        with pytest.raises(ImportError, match="transformers"):
            mod.HuggingFaceTranslator(
                model_id="any/model", runs_dir="/tmp/x", run_name="x",
            )

    def test_error_message_includes_install_hint(self):
        mod = _reload_hf_module()
        with pytest.raises(ImportError, match=r"\[hf-models\]"):
            mod.HuggingFaceTranslator(
                model_id="any/model", runs_dir="/tmp/x", run_name="x",
            )


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="requires the 'transformers' package",
)
class TestInstantiationWithTransformers:
    def test_constructor_does_not_download(self):
        """Constructor must be lazy — instantiating shouldn't hit the Hub."""
        from autonmt.backends.huggingface.translation_engine import HuggingFaceTranslator
        # If the constructor tried to download anything, this would fail on
        # offline CI. The model is bogus on purpose.
        trans = HuggingFaceTranslator(
            model_id="this-does-not-exist/never-downloaded",
            src_lang="de", tgt_lang="en",
            runs_dir="/tmp/hf_test", run_name="test_lazy",
        )
        assert trans._model is None
        assert trans._tokenizer is None
        assert trans.ENGINE == "huggingface"

    def test_checkpoint_gate_skips_when_finetuned_exists(self, tmp_path):
        """If a HF checkpoint (config.json) already lives at the path and
        force_overwrite is False, _train must skip and repoint model_id."""
        from autonmt.backends.huggingface.translation_engine import HuggingFaceTranslator
        # Simulate a previous fine-tune: just drop a config.json there.
        ckpt = tmp_path / "checkpoints"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")

        trans = HuggingFaceTranslator(
            model_id="any/model", src_lang="de", tgt_lang="en",
            runs_dir=str(tmp_path), run_name="test_gate",
        )
        # Should return immediately without loading model or calling HF Trainer.
        trans._train(train_ds=None, checkpoints_dir=str(ckpt),
                     logs_path=str(tmp_path / "logs"), force_overwrite=False)
        assert trans._model is None  # never loaded
        assert trans.model_id == str(ckpt)
        assert trans.tokenizer_id == str(ckpt)

    def test_from_dataset_autofills_langs(self):
        from autonmt.backends.huggingface.translation_engine import HuggingFaceTranslator
        # Stub a Dataset-like object with just the attributes from_dataset needs.
        fake_ds = SimpleNamespace(
            src_lang="de", tgt_lang="en",
            get_runs_path=lambda toolkit: f"/tmp/runs/{toolkit}",
            get_run_name=lambda run_prefix: f"{run_prefix}_test",
        )
        trans = HuggingFaceTranslator.from_dataset(
            fake_ds, model_id="any/model", run_prefix="hf-base",
        )
        assert trans.src_lang == "de"
        assert trans.tgt_lang == "en"
        assert trans.runs_dir == "/tmp/runs/huggingface"
        assert trans.run_name == "hf-base_test"

    def test_from_dataset_explicit_lang_wins(self):
        """If the user passes src_lang / tgt_lang explicitly, they override the dataset."""
        from autonmt.backends.huggingface.translation_engine import HuggingFaceTranslator
        fake_ds = SimpleNamespace(
            src_lang="de", tgt_lang="en",
            get_runs_path=lambda toolkit: f"/tmp/runs/{toolkit}",
            get_run_name=lambda run_prefix: f"{run_prefix}_test",
        )
        trans = HuggingFaceTranslator.from_dataset(
            fake_ds, model_id="any/model", run_prefix="hf-base",
            src_lang="fr", tgt_lang="es",
        )
        assert trans.src_lang == "fr"
        assert trans.tgt_lang == "es"

    def test_training_args_mapping(self, tmp_path):
        """FitConfig fields map to the Seq2SeqTrainingArguments kwargs dict."""
        from autonmt.backends.huggingface.translation_engine import HuggingFaceTranslator
        trans = HuggingFaceTranslator(
            model_id="any/model", src_lang="de", tgt_lang="en",
            runs_dir=str(tmp_path), run_name="test_args",
        )
        fit_kwargs = dict(
            max_epochs=2, batch_size=16, learning_rate=3e-5,
            weight_decay=0.01, gradient_clip_val=0.5,
            accumulate_grad_batches=2, save_best=True,
            monitor="eval_loss", seed=7, num_workers=4,
        )
        mapped = trans._build_training_args_dict(
            fit_kwargs=fit_kwargs,
            output_dir=tmp_path / "ckpt", logs_dir=tmp_path / "logs",
            force_overwrite=False,
        )
        assert mapped["num_train_epochs"] == 2
        assert mapped["per_device_train_batch_size"] == 16
        assert mapped["learning_rate"] == 3e-5
        assert mapped["weight_decay"] == 0.01
        assert mapped["max_grad_norm"] == 0.5
        assert mapped["gradient_accumulation_steps"] == 2
        assert mapped["load_best_model_at_end"] is True
        assert mapped["metric_for_best_model"] == "eval_loss"
        assert mapped["seed"] == 7
        assert mapped["dataloader_num_workers"] == 4

    def test_training_args_hf_overrides_win(self, tmp_path):
        """hf_training_args=dict(...) wins over the mapped values on collision.

        The collision warning is logged via the autonmt logger; verifying its
        presence from a test isn't worth the gymnastics (the logger has
        propagate=False and its handler captures stderr at configure time,
        before pytest's capsys hooks in). We assert on the behaviour instead.
        """
        from autonmt.backends.huggingface.translation_engine import HuggingFaceTranslator
        trans = HuggingFaceTranslator(
            model_id="any/model", src_lang="de", tgt_lang="en",
            runs_dir=str(tmp_path), run_name="test_overrides",
        )
        fit_kwargs = dict(
            max_epochs=2, batch_size=8, learning_rate=5e-5,
            hf_training_args={"learning_rate": 1e-4, "label_smoothing_factor": 0.1},
        )
        mapped = trans._build_training_args_dict(
            fit_kwargs=fit_kwargs,
            output_dir=tmp_path / "ckpt", logs_dir=tmp_path / "logs",
            force_overwrite=False,
        )
        # Override wins.
        assert mapped["learning_rate"] == 1e-4
        # HF-only kwarg passes through (if the installed transformers accepts it).
        assert mapped.get("label_smoothing_factor") == 0.1

    def test_build_run_report_without_loaded_model(self):
        """Report builder must tolerate an unloaded model + missing AutoNMT vocab.

        Regression: the parent ``parse_metrics`` used to crash on ``self.model``
        and ``self.src_vocab`` for HF runs. The override on HF builds the same
        schema sourced from the HF tokenizer + model id.
        """
        from autonmt.backends.huggingface.translation_engine import HuggingFaceTranslator
        trans = HuggingFaceTranslator(
            model_id="dummy/model", src_lang="de", tgt_lang="en",
            runs_dir="/tmp/hf_test", run_name="rep_test",
        )
        fake_eval = SimpleNamespace(
            src_lang="de", tgt_lang="en", dataset_name="multi30k",
        )
        # No fit / no translate happened — _model / _tokenizer are still None.
        report = trans._build_run_report(eval_ds=fake_eval, translations={"beam1": {}})
        assert report["engine"] == "huggingface"
        assert report["model__architecture"] == "huggingface:dummy/model"
        assert report["vocab__subword_model"] == "hf_tokenizer"
        assert report["vocab__lang_pair"] == "de-en"
        assert report["test_dataset"] == "multi30k"
        assert report["translations"] == {"beam1": {}}
