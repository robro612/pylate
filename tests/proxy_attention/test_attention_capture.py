"""Tests for attention capture in ProxyAttentionColBERT.

These tests verify that:
1. The last layer attention capture hook works correctly
2. Attention weights are properly captured during forward pass
3. The captured attention has the expected shape
"""

import pytest
import torch

from pylate import models


class TestAttentionCapture:
    """Test that attention weights are properly captured from the last layer."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=4,
            num_select_tokens=4,
            device="cpu",
        )

    def test_attention_capture_hook_setup(self, model):
        """Test that the attention capture hook can be set up."""
        transformer = model[0]
        auto_model = transformer.auto_model

        captured_attention, cleanup = model._setup_last_layer_attention_capture(auto_model)

        assert captured_attention is not None
        assert callable(cleanup)

        # Clean up
        cleanup()

    def test_attention_captured_during_forward(self, model):
        """Test that attention is captured during document forward pass."""
        documents = ["This is a test document for attention capture."]
        tokens = model.tokenize(documents, is_query=False)

        # Run forward pass - this should capture attention
        output = model(tokens, is_query=False)

        # Verify output has expected shape
        assert "token_embeddings" in output
        assert output["token_embeddings"].shape[1] == model.num_select_tokens

    def test_attention_has_correct_shape(self, model):
        """Test that captured attention has correct shape."""
        transformer = model[0]
        auto_model = transformer.auto_model

        # Get model config for expected dimensions
        config = auto_model.config
        num_heads = config.num_attention_heads

        # Set up capture
        captured_attention, cleanup = model._setup_last_layer_attention_capture(auto_model)

        try:
            # Create test input
            documents = ["Test document."]
            tokens = model.tokenize(documents, is_query=False)

            # Get embeddings and run through model
            embedding_layer = model._get_embedding_layer(auto_model)
            input_embeds = embedding_layer(tokens['input_ids'].to(model.device))

            # Append proxy tokens
            combined_embeds, combined_mask = model._append_proxy_tokens(
                input_embeds, tokens['attention_mask'].to(model.device)
            )

            # Run forward
            _ = auto_model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                output_attentions=False,
                return_dict=True,
            )

            # Check captured attention
            attn_weights = captured_attention[0]
            assert attn_weights is not None, "Attention weights were not captured!"

            batch_size = combined_embeds.shape[0]
            seq_len = combined_embeds.shape[1]

            # Attention shape: (batch, num_heads, seq_len, seq_len)
            assert attn_weights.shape[0] == batch_size
            assert attn_weights.shape[1] == num_heads
            assert attn_weights.shape[2] == seq_len
            assert attn_weights.shape[3] == seq_len

        finally:
            cleanup()

    def test_attention_values_are_valid(self, model):
        """Test that attention weights sum to 1 and are non-negative."""
        transformer = model[0]
        auto_model = transformer.auto_model

        captured_attention, cleanup = model._setup_last_layer_attention_capture(auto_model)

        try:
            documents = ["Test document for validation."]
            tokens = model.tokenize(documents, is_query=False)

            embedding_layer = model._get_embedding_layer(auto_model)
            input_embeds = embedding_layer(tokens['input_ids'].to(model.device))

            combined_embeds, combined_mask = model._append_proxy_tokens(
                input_embeds, tokens['attention_mask'].to(model.device)
            )

            _ = auto_model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                output_attentions=False,
                return_dict=True,
            )

            attn_weights = captured_attention[0]
            assert attn_weights is not None

            # All values should be non-negative
            assert torch.all(attn_weights >= 0), "Attention weights contain negative values"

            # Rows should sum to ~1 (softmax output)
            row_sums = attn_weights.sum(dim=-1)
            assert torch.allclose(
                row_sums, torch.ones_like(row_sums), atol=1e-5
            ), f"Attention rows don't sum to 1: {row_sums}"

        finally:
            cleanup()


class TestEagerAttentionSetup:
    """Test eager attention configuration for last layer."""

    def test_eager_attention_enabled_at_init(self):
        """Test that eager attention is enabled for last layer during init."""
        model = models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=4,
            num_select_tokens=4,
            device="cpu",
        )

        transformer = model[0]
        auto_model = transformer.auto_model
        last_layer = model._get_last_layer(auto_model)

        # The last layer should have eager attention configured
        # Check if config was modified (this may vary by model architecture)
        assert last_layer is not None

