"""Integration tests for ProxyAttentionColBERT training loop."""

import pytest
import torch
from torch.optim import AdamW

from pylate import models, losses


class TestProxyAttentionTrainingLoop:
    """Test a complete training iteration."""

    @pytest.fixture
    def model(self):
        model = models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=8,
            num_select_tokens=8,
            use_cluster_pooling=True,
            device="cpu",
        )
        model.train()
        return model

    def test_single_training_step(self, model):
        """Test a single training step with optimizer."""
        loss_fn = losses.ProxyAttentionDistillation(model=model)
        optimizer = AdamW(model.parameters(), lr=1e-5)
        
        queries = ["What is machine learning?"]
        documents = ["Machine learning is AI.", "Deep learning is a subset."]
        
        query_tokens = model.tokenize(queries, is_query=True)
        doc_tokens = model.tokenize(documents, is_query=False)
        teacher_scores = torch.tensor([[0.9, 0.1]])
        
        # Store initial weights
        initial_dense_weight = model[1].linear.weight.clone()
        initial_proxy_weight = model._proxy_embeddings.weight.clone()
        
        # Training step
        optimizer.zero_grad()
        loss = loss_fn([query_tokens, doc_tokens], teacher_scores)
        loss.backward()
        optimizer.step()
        
        # Verify weights changed
        assert not torch.allclose(initial_dense_weight, model[1].linear.weight), \
            "Dense weights should change after training step"
        assert not torch.allclose(initial_proxy_weight, model._proxy_embeddings.weight), \
            "Proxy weights should change after training step"

    def test_multiple_training_steps(self, model):
        """Test multiple training steps."""
        loss_fn = losses.ProxyAttentionDistillation(model=model)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        queries = ["What is machine learning?", "How does AI work?"]
        documents = [
            "Machine learning is AI.", "Deep learning is a subset.",
            "AI uses algorithms.", "Neural networks are powerful.",
        ]
        teacher_scores = torch.tensor([[0.9, 0.1], [0.3, 0.7]])
        
        losses_history = []
        for _ in range(3):
            query_tokens = model.tokenize(queries, is_query=True)
            doc_tokens = model.tokenize(documents, is_query=False)
            
            optimizer.zero_grad()
            loss = loss_fn([query_tokens, doc_tokens], teacher_scores)
            loss.backward()
            optimizer.step()
            
            losses_history.append(loss.item())
        
        # Loss should generally decrease (though not strictly guaranteed)
        assert len(losses_history) == 3
        # At least check all losses are valid numbers
        for l in losses_history:
            assert l >= 0 and not torch.isnan(torch.tensor(l))


class TestProxyAttentionScoring:
    """Test scoring/inference functionality."""

    @pytest.fixture
    def model(self):
        return models.ProxyAttentionColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            num_proxy_tokens=8,
            num_select_tokens=8,
            use_cluster_pooling=True,
            device="cpu",
        )

    def test_encode_queries(self, model):
        """Test encoding queries."""
        queries = ["What is machine learning?", "How does neural network work?"]
        tokens = model.tokenize(queries, is_query=True)
        
        with torch.no_grad():
            output = model(tokens, is_query=True)
        
        embeddings = output["token_embeddings"]
        assert embeddings.shape[0] == 2  # 2 queries

    def test_encode_documents(self, model):
        """Test encoding documents with proxy attention."""
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are computing systems inspired by the brain.",
            "Deep learning uses multiple layers.",
        ]
        tokens = model.tokenize(documents, is_query=False)
        
        with torch.no_grad():
            output = model(tokens, is_query=False)
        
        embeddings = output["token_embeddings"]
        assert embeddings.shape[0] == 3  # 3 documents
        assert embeddings.shape[1] == 8  # num_select_tokens

    def test_colbert_scoring(self, model):
        """Test ColBERT-style scoring between queries and documents."""
        from pylate.scores import colbert_kd_scores
        
        queries = ["What is machine learning?"]
        documents = ["Machine learning is AI.", "Neural networks are inspired by the brain."]
        
        query_tokens = model.tokenize(queries, is_query=True)
        doc_tokens = model.tokenize(documents, is_query=False)
        
        with torch.no_grad():
            query_output = model(query_tokens, is_query=True)
            doc_output = model(doc_tokens, is_query=False)
        
        query_embeddings = torch.nn.functional.normalize(
            query_output["token_embeddings"], p=2, dim=-1
        )
        doc_embeddings = torch.nn.functional.normalize(
            doc_output["token_embeddings"], p=2, dim=-1
        )
        
        # Reshape for kd_scores: (batch, n_docs, tokens, dim)
        doc_embeddings = doc_embeddings.view(1, 2, *doc_embeddings.shape[1:])
        
        scores = colbert_kd_scores(query_embeddings, doc_embeddings)
        
        assert scores.shape == (1, 2)  # 1 query, 2 documents
        # First doc should score higher (more relevant)
        assert scores[0, 0] > scores[0, 1], "Relevant doc should score higher"

