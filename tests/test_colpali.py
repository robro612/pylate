from __future__ import annotations

import pytest

from pylate import models
from pylate.models.colbert import (
    _COLPALI_TO_BASE_ARCHITECTURE,
)

COLPALI_BASE = "vidore/colqwen2.5-base"
COLPALI_ADAPTER = "vidore/colqwen2.5-v0.2"


class TestDetectColpaliArchitecture:
    def test_detects_colpali_base_model(self):
        """Base ColPali repo has config.json with ColQwen2_5 architecture."""
        arch, config = models.ColBERT._detect_colpali_architecture(COLPALI_BASE, {})
        assert arch is not None
        assert arch in _COLPALI_TO_BASE_ARCHITECTURE
        assert config is not None
        assert config.model_type == "qwen2_5_vl"

    def test_adapter_repo_detected(self):
        """LoRA adapter repos resolve via adapter_config.json -> base model config."""
        arch, config = models.ColBERT._detect_colpali_architecture(COLPALI_ADAPTER, {})
        assert arch is not None
        assert arch in _COLPALI_TO_BASE_ARCHITECTURE
        assert config is not None

    def test_non_colpali_model(self):
        arch, config = models.ColBERT._detect_colpali_architecture(
            "bert-base-uncased", {}
        )
        assert arch is None
        assert config is not None

    def test_invalid_model_path(self):
        arch, config = models.ColBERT._detect_colpali_architecture(
            "this-model-does-not-exist-12345", {}
        )
        assert arch is None
        assert config is None


class TestColpaliBaseModelLoading:
    """Test loading a non-adapter ColPali checkpoint (has config.json)."""

    @pytest.fixture(scope="class")
    def colpali_model(self):
        return models.ColBERT(
            model_name_or_path=COLPALI_BASE,
            device="cpu",
        )

    def test_is_colpali_flag(self, colpali_model):
        assert colpali_model._is_colpali_model is True

    def test_has_two_modules(self, colpali_model):
        module_list = list(colpali_model)
        assert len(module_list) == 2

    def test_dense_projection_dimensions(self, colpali_model):
        module_list = list(colpali_model)
        dense = module_list[1]
        assert dense.in_features > 0
        assert dense.out_features == 128

    def test_encode_text_query(self, colpali_model):
        embeddings = colpali_model.encode(["what is machine learning?"], is_query=True)
        assert len(embeddings) == 1
        assert embeddings[0].ndim == 2
        assert embeddings[0].shape[-1] == 128

    def test_processor_is_base_vlm(self, colpali_model):
        """The processor should be the base VLM processor, not a ColPali one."""
        proc_name = type(colpali_model._first_module().processor).__name__
        assert "Col" not in proc_name


class TestColpaliAdapterLoading:
    """Load a LoRA-only ColPali checkpoint (adapter_model.safetensors, no merged
    weights). The ``custom_text_proj`` projection must be reconstructed by
    merging the base repo's frozen projection with the adapter's LoRA pair."""

    @pytest.fixture(scope="class")
    def adapter_model(self):
        return models.ColBERT(model_name_or_path=COLPALI_ADAPTER, device="cpu")

    def test_is_colpali_flag(self, adapter_model):
        assert adapter_model._is_colpali_model is True

    def test_dense_projection_dimensions(self, adapter_model):
        dense = list(adapter_model)[1]
        assert dense.in_features > 0
        assert dense.out_features == 128

    def test_encode_text_query(self, adapter_model):
        embeddings = adapter_model.encode(["what is machine learning?"], is_query=True)
        assert len(embeddings) == 1
        assert embeddings[0].shape[-1] == 128


class TestColpaliProjectionMerge:
    """The LoRA merge must actually apply the adapter delta, not silently fall
    back to the (frozen) base projection."""

    def test_lora_delta_is_applied(self):
        import torch

        from pylate.models.colbert import (
            _merge_lora_proj_tensors,
            _read_merged_proj_tensors,
        )

        weight_key, bias_key = "custom_text_proj.weight", "custom_text_proj.bias"
        base = _read_merged_proj_tensors(COLPALI_BASE, weight_key, bias_key, {})
        merged = _merge_lora_proj_tensors(COLPALI_ADAPTER, weight_key, bias_key, {})

        assert weight_key in base
        assert weight_key in merged
        delta = merged[weight_key].float() - base[weight_key].float()
        assert torch.linalg.norm(delta) > 1e-3
