"""FitConfig / PredictConfig precedence: explicit kwargs > config > defaults."""
import pytest

from autonmt.backends._base.config import FitConfig, PredictConfig, UNSET, merge_config


class TestMerge:
    def test_defaults_apply_when_nothing_passed(self):
        cfg, extra = merge_config(None, FitConfig, {})
        assert cfg["batch_size"] == 128
        assert cfg["max_epochs"] == 1
        assert extra == {}

    def test_legacy_kwargs_without_config(self):
        cfg, extra = merge_config(None, FitConfig, {"batch_size": 32, "optimizer": "sgd"})
        assert cfg["batch_size"] == 32
        assert cfg["optimizer"] == "sgd"

    def test_config_values_are_used(self):
        cfg, extra = merge_config(
            FitConfig(batch_size=64, max_epochs=10), FitConfig,
            {"batch_size": UNSET, "max_epochs": UNSET},
        )
        assert cfg["batch_size"] == 64
        assert cfg["max_epochs"] == 10

    def test_explicit_kwarg_overrides_config(self):
        cfg, extra = merge_config(
            FitConfig(batch_size=64, max_epochs=10), FitConfig,
            {"batch_size": 256, "max_epochs": UNSET},
        )
        assert cfg["batch_size"] == 256  # explicit wins
        assert cfg["max_epochs"] == 10   # from config

    def test_unknown_kwargs_routed_to_extras(self):
        cfg, extra = merge_config(
            None, FitConfig,
            {"batch_size": 32, "wandb_params": {"project": "x"}},
        )
        assert cfg["batch_size"] == 32
        assert extra == {"wandb_params": {"project": "x"}}

    def test_unset_extras_are_dropped(self):
        cfg, extra = merge_config(None, FitConfig, {"unrelated": UNSET})
        assert extra == {}

    def test_predict_config_independent(self):
        cfg, extra = merge_config(PredictConfig(beams=[1, 5]), PredictConfig, {})
        assert cfg["beams"] == [1, 5]
        assert cfg["eval_mode"] == "same"

    def test_predict_config_decoder_round_trips(self):
        """A BaseSearch instance passed via PredictConfig must survive ``merge_config``
        so the translator can pick it up. ``asdict`` deep-copies the value, so we
        check type + attrs rather than identity."""
        from autonmt.core.decoding import TopPSampling
        decoder = TopPSampling(top_p=0.85, temperature=0.7)
        cfg, _ = merge_config(PredictConfig(decoder=decoder), PredictConfig, {})
        assert isinstance(cfg["decoder"], TopPSampling)
        assert cfg["decoder"].top_p == 0.85
        assert cfg["decoder"].temperature == 0.7

    def test_predict_config_decoder_kwarg_overrides_config(self):
        from autonmt.core.decoding import GreedySearch, TopKSampling
        cfg, _ = merge_config(
            PredictConfig(decoder=GreedySearch()), PredictConfig,
            {"decoder": TopKSampling(top_k=5)},
        )
        assert isinstance(cfg["decoder"], TopKSampling)

    def test_wrong_config_type_raises(self):
        with pytest.raises(TypeError, match="FitConfig"):
            merge_config(PredictConfig(), FitConfig, {})
