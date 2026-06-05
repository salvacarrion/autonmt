"""Contract tests for the HuggingFace LM backends (no Hub download, no accelerate).

The parts that need a real model (loading, fit, generate, fill_mask) are exercised
by users with `transformers` + `accelerate` installed; here we verify the wiring
that doesn't touch the network: lazy exports, packing, the causal collator,
`from_corpus`, and the FitConfig → TrainingArguments mapping.
"""
import importlib.util

import pytest
import torch

from autonmt.backends.huggingface.lm import _CausalCollator, _HuggingFaceLMBase
from autonmt.datasets.lm_corpus import LMCorpus

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def test_lazy_export_from_backends():
    import autonmt.backends as backends
    assert backends.HuggingFaceCausalLM.__name__ == "HuggingFaceCausalLM"
    assert backends.HuggingFaceMaskedLM.__name__ == "HuggingFaceMaskedLM"
    assert backends.HuggingFaceCausalLM.TASK == "causal"
    assert backends.HuggingFaceMaskedLM.TASK == "masked"
    assert backends.HuggingFaceCausalLM.ENGINE == "huggingface"


def test_group_blocks_packs_and_drops_remainder():
    blocks = _HuggingFaceLMBase._group_blocks([[1, 2, 3], [4, 5, 6, 7], [8, 9]], block_size=4)
    assert blocks == [[1, 2, 3, 4], [5, 6, 7, 8]]   # last token (9) dropped


def test_group_blocks_too_small_raises():
    with pytest.raises(ValueError, match="too small"):
        _HuggingFaceLMBase._group_blocks([[1, 2]], block_size=8)


def test_causal_collator_pads_labels_with_ignore():
    col = _CausalCollator(pad_token_id=0)
    batch = col([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
                 {"input_ids": [8, 9], "labels": [8, 9]}])
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 9, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [8, 9, -100]]  # pad → -100
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["input_ids"].dtype == torch.long


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_from_corpus_wires_run_layout():
    from autonmt.backends import HuggingFaceCausalLM
    corpus = LMCorpus(base_path="/tmp/x", name="wiki", size_name="original",
                      mode="text", subword_model="bpe", vocab_size=16000)
    m = HuggingFaceCausalLM.from_corpus(corpus, run_prefix="ft", model_id="gpt2", device="cpu")
    assert m.run_name == "ft_bpe_16000"
    assert m.model_id == "gpt2"
    assert "models/huggingface/runs/ft_bpe_16000/checkpoints" in m.get_model_checkpoints_path()


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_fit_config_maps_to_training_arguments():
    from autonmt.backends import HuggingFaceMaskedLM
    from autonmt.backends._base.config import FitConfig
    m = HuggingFaceMaskedLM(model_id="bert-base-uncased", device="cpu")
    mapped = m._build_training_args_dict(
        FitConfig(max_epochs=3, batch_size=16, learning_rate=5e-5).as_kwargs(),
        output_dir="/out", logs_dir="/logs", force_overwrite=False)
    assert mapped["num_train_epochs"] == 3
    assert mapped["per_device_train_batch_size"] == 16
    assert mapped["learning_rate"] == 5e-5
    # Every mapped key must be a real TrainingArguments parameter (unsupported dropped).
    from transformers import TrainingArguments
    import inspect
    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    assert set(mapped).issubset(valid)
    # fp32 default sets no half-precision flag.
    assert mapped.get("fp16") is not True and mapped.get("bf16") is not True


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_precision_maps_to_hf_flag():
    from autonmt.backends import HuggingFaceCausalLM
    from autonmt.backends._base.config import FitConfig
    m = HuggingFaceCausalLM(model_id="gpt2", device="cpu")
    mapped = m._build_training_args_dict(
        FitConfig(max_epochs=1, precision="bf16").as_kwargs(),
        output_dir="/out", logs_dir="/logs", force_overwrite=False)
    assert mapped.get("bf16") is True
    assert mapped.get("fp16") is not True
