"""Tests for ProxyAttentionDistillation loss function."""

import pytest
import torch

from pylate import models, losses


class TestProxyAttentionDistillationInit:
    """Test loss function initialization."""

    def test_basic_init(self):
        """Test basic loss initialization."""
        model = models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=8,
            num_select_tokens=8,
            device="cpu",
        )
        loss_fn = losses.ProxyAttentionDistillation(model=model)
        
        assert loss_fn.model is model
        assert loss_fn.normalize_scores is False  # Default is False to preserve gradients

    def test_init_with_normalize_scores(self):
        """Test initialization with normalize_scores parameter."""
        model = models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=8,
            num_select_tokens=8,
            device="cpu",
        )
        loss_fn = losses.ProxyAttentionDistillation(
            model=model, normalize_scores=True
        )
        assert loss_fn.normalize_scores is True


class TestProxyAttentionDistillationForward:
    """Test loss function forward pass."""

    @pytest.fixture
    def model(self):
        return models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=8,
            num_select_tokens=8,
            use_cluster_pooling=True,
            device="cpu",
        )

    def test_forward_basic(self, model):
        """Test basic forward pass."""
        loss_fn = losses.ProxyAttentionDistillation(model=model)
        
        queries = ["What is machine learning?"]
        # 2 documents per query
        documents = ["Machine learning is AI.", "Deep learning is a subset."]
        
        query_tokens = model.tokenize(queries, is_query=True)
        doc_tokens = model.tokenize(documents, is_query=False)
        
        # Teacher scores: (num_queries, num_docs_per_query)
        teacher_scores = torch.tensor([[0.9, 0.1]])
        
        loss = loss_fn([query_tokens, doc_tokens], teacher_scores)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # KL divergence is non-negative

    def test_forward_multiple_queries(self, model):
        """Test forward with multiple queries."""
        loss_fn = losses.ProxyAttentionDistillation(model=model)
        
        queries = ["What is machine learning?", "How does neural network work?"]
        # 2 documents per query = 4 total
        documents = [
            "Machine learning is AI.",
            "Neural networks are inspired by the brain.",
            "Deep learning uses multiple layers.",
            "AI is transforming industries.",
        ]
        
        query_tokens = model.tokenize(queries, is_query=True)
        doc_tokens = model.tokenize(documents, is_query=False)
        
        # Teacher scores: (2 queries, 2 docs per query)
        teacher_scores = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        
        loss = loss_fn([query_tokens, doc_tokens], teacher_scores)
        
        assert loss.item() >= 0


class TestProxyAttentionDistillationGradients:
    """Test that gradients flow correctly."""

    @pytest.fixture
    def model(self):
        model = models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=8,
            num_select_tokens=8,
            use_cluster_pooling=False,
            device="cpu",
        )
        model.train()
        return model

    def test_gradients_flow(self, model):
        """Test that gradients flow through the model."""
        loss_fn = losses.ProxyAttentionDistillation(model=model)
        
        queries = ["What is machine learning?"]
        documents = ["Machine learning is AI.", "Deep learning is a subset."]
        
        query_tokens = model.tokenize(queries, is_query=True)
        doc_tokens = model.tokenize(documents, is_query=False)
        teacher_scores = torch.tensor([[0.9, 0.1]])
        
        model.zero_grad()
        loss = loss_fn([query_tokens, doc_tokens], teacher_scores)
        loss.backward()
        
        # Check that gradients exist and are non-zero
        dense_grad = model[1].linear.weight.grad
        assert dense_grad is not None
        assert dense_grad.norm().item() > 0, "Dense layer should have non-zero gradients"
        
        proxy_grad = model._proxy_embeddings.weight.grad
        assert proxy_grad is not None
        assert proxy_grad.norm().item() > 0, "Proxy embeddings should have non-zero gradients"

    def test_gradients_with_cluster_pooling(self):
        """Test gradients with cluster pooling enabled."""
        model = models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=8,
            num_select_tokens=8,
            use_cluster_pooling=True,
            device="cpu",
        )
        model.train()
        loss_fn = losses.ProxyAttentionDistillation(model=model)
        
        queries = ["What is machine learning?"]
        documents = ["Machine learning is AI.", "Deep learning is a subset."]
        
        query_tokens = model.tokenize(queries, is_query=True)
        doc_tokens = model.tokenize(documents, is_query=False)
        teacher_scores = torch.tensor([[0.9, 0.1]])
        
        model.zero_grad()
        loss = loss_fn([query_tokens, doc_tokens], teacher_scores)
        loss.backward()
        
        dense_grad = model[1].linear.weight.grad
        assert dense_grad is not None
        assert dense_grad.norm().item() > 0

