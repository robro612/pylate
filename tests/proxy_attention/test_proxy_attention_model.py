"""Tests for ProxyAttentionColBERT model initialization and forward pass."""

import pytest
import torch
import tempfile
import os

from pylate import models


class TestProxyAttentionColBERTInit:
    """Test model initialization."""

    def test_basic_init(self):
        """Test basic model initialization."""
        model = models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=8,
            num_select_tokens=8,
            device="cpu",
        )
        assert model.num_proxy_tokens == 8
        assert model.num_select_tokens == 8
        assert model._proxy_embeddings is not None
        assert model._proxy_embeddings.weight.shape[0] == 8

    def test_init_with_cluster_pooling(self):
        """Test initialization with cluster pooling enabled."""
        model = models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=16,
            num_select_tokens=8,
            use_cluster_pooling=True,
            cluster_centroid_weight=2.0,
            device="cpu",
        )
        assert model.use_cluster_pooling is True
        assert model.cluster_centroid_weight == 2.0
        assert model.num_proxy_tokens == 16
        assert model.num_select_tokens == 8

    def test_init_different_proxy_select_tokens(self):
        """Test that num_proxy_tokens can be different from num_select_tokens."""
        model = models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=32,
            num_select_tokens=16,
            device="cpu",
        )
        assert model.num_proxy_tokens == 32
        assert model.num_select_tokens == 16


class TestProxyAttentionColBERTTokenize:
    """Test tokenization."""

    @pytest.fixture
    def model(self):
        return models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=8,
            num_select_tokens=8,
            device="cpu",
        )

    def test_query_tokenization(self, model):
        """Test query tokenization."""
        queries = ["What is machine learning?"]
        tokens = model.tokenize(queries, is_query=True)
        
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape[0] == 1  # batch size

    def test_document_tokenization(self, model):
        """Test document tokenization."""
        documents = ["Machine learning is AI.", "Deep learning is a subset."]
        tokens = model.tokenize(documents, is_query=False)
        
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape[0] == 2  # batch size


class TestProxyAttentionColBERTForward:
    """Test forward pass."""

    @pytest.fixture
    def model(self):
        return models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=8,
            num_select_tokens=8,
            use_cluster_pooling=False,
            device="cpu",
        )

    def test_query_forward(self, model):
        """Test query forward pass."""
        queries = ["What is machine learning?"]
        tokens = model.tokenize(queries, is_query=True)
        output = model(tokens, is_query=True)
        
        assert "token_embeddings" in output
        # Query embeddings should have original sequence length
        assert output["token_embeddings"].shape[0] == 1

    def test_document_forward(self, model):
        """Test document forward pass with proxy attention."""
        documents = ["Machine learning is AI.", "Deep learning is a subset."]
        tokens = model.tokenize(documents, is_query=False)
        output = model(tokens, is_query=False)
        
        assert "token_embeddings" in output
        # Document embeddings should have fixed size (num_select_tokens)
        assert output["token_embeddings"].shape[0] == 2
        assert output["token_embeddings"].shape[1] == 8  # num_select_tokens

    def test_document_forward_with_cluster_pooling(self):
        """Test document forward with cluster pooling."""
        model = models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=8,
            num_select_tokens=8,
            use_cluster_pooling=True,
            device="cpu",
        )
        documents = ["Machine learning is a subset of artificial intelligence."]
        tokens = model.tokenize(documents, is_query=False)
        output = model(tokens, is_query=False)
        
        assert output["token_embeddings"].shape[1] == 8

    def test_forward_requires_grad(self, model):
        """Test that forward pass preserves gradients."""
        model.train()
        documents = ["Machine learning is AI."]
        tokens = model.tokenize(documents, is_query=False)
        output = model(tokens, is_query=False)
        
        assert output["token_embeddings"].requires_grad is True


class TestProxyAttentionColBERTSaveLoad:
    """Test model save/load functionality."""

    def test_save_and_load(self):
        """Test that model can be saved and loaded."""
        model = models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=8,
            num_select_tokens=8,
            use_cluster_pooling=True,
            device="cpu",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model")
            model.save(save_path)
            
            loaded_model = models.ProxyAttentionColBERT.load(save_path, device="cpu")
            
            # Verify config
            assert loaded_model.num_proxy_tokens == model.num_proxy_tokens
            assert loaded_model.num_select_tokens == model.num_select_tokens
            assert loaded_model.use_cluster_pooling == model.use_cluster_pooling
            
            # Verify proxy embeddings
            original_proxy = model._proxy_embeddings.weight.detach()
            loaded_proxy = loaded_model._proxy_embeddings.weight.detach()
            assert torch.allclose(original_proxy, loaded_proxy)

