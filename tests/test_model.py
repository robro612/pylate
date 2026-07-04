from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pylate import models, rank
from pylate.models import colbert


@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_model_creation(**kwargs) -> None:
    """Test the creation of different models."""
    query = ["fruits are healthy."]
    documents = [["fruits are healthy.", "fruits are good for health."]]
    torch.manual_seed(42)
    # Creation from a base encoder
    model = models.ColBERT(model_name_or_path="bert-base-uncased")
    # We don't test the embeddings of newly initied models for now as we need to make it deterministic
    # queries_embeddings = model.encode(sentences=query, is_query=True)
    # documents_embeddings = model.encode(sentences=documents, is_query=False)
    # reranked_documents = rank.rerank(
    #     documents_ids=[["1", "2"]],
    #     queries_embeddings=queries_embeddings,
    #     documents_embeddings=documents_embeddings,
    # )
    # assert math.isclose(
    #     reranked_documents[0][0]["score"], 25.92, rel_tol=0.01, abs_tol=0.01
    # )
    # assert math.isclose(reranked_documents[0][1]["score"], 23.7, rel_tol=0.01, abs_tol=0.01)

    # Creation from a base sentence-transformer
    model = models.ColBERT(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
    # We don't test the embeddings of newly initied models for now as we need to make it deterministic
    # queries_embeddings = model.encode(sentences=query, is_query=True)
    # documents_embeddings = model.encode(sentences=documents, is_query=False)
    # reranked_documents = rank.rerank(
    #     documents_ids=[["1", "2"]],
    #     queries_embeddings=queries_embeddings,
    #     documents_embeddings=documents_embeddings,
    # )
    # assert math.isclose(
    #     reranked_documents[0][0]["score"], 18.77, rel_tol=0.01, abs_tol=0.01
    # )
    # assert math.isclose(
    #     reranked_documents[0][1]["score"], 18.63, rel_tol=0.01, abs_tol=0.01

    # Creation from stanford-nlp (safetensor)
    model = models.ColBERT(model_name_or_path="answerdotai/answerai-colbert-small-v1")
    queries_embeddings = model.encode(sentences=query, is_query=True)
    documents_embeddings = model.encode(sentences=documents, is_query=False)
    reranked_documents = rank.rerank(
        documents_ids=[["1", "2"]],
        queries_embeddings=queries_embeddings,
        documents_embeddings=documents_embeddings,
    )
    assert math.isclose(
        reranked_documents[0][0]["score"], 31.71, rel_tol=0.01, abs_tol=0.01
    )
    assert math.isclose(
        reranked_documents[0][1]["score"], 31.64, rel_tol=0.01, abs_tol=0.01
    )

    # Creation from stanford-nlp (bin)
    model = models.ColBERT(model_name_or_path="Crystalcareai/Colbertv2")
    queries_embeddings = model.encode(sentences=query, is_query=True)
    documents_embeddings = model.encode(sentences=documents, is_query=False)
    reranked_documents = rank.rerank(
        documents_ids=[["1", "2"]],
        queries_embeddings=queries_embeddings,
        documents_embeddings=documents_embeddings,
    )
    assert math.isclose(
        reranked_documents[0][0]["score"], 31.15, rel_tol=0.01, abs_tol=0.01
    )
    assert math.isclose(
        reranked_documents[0][1]["score"], 30.61, rel_tol=0.01, abs_tol=0.01
    )

    # Creation from PyLate
    model = models.ColBERT(model_name_or_path="lightonai/colbertv2.0")
    queries_embeddings = model.encode(sentences=query, is_query=True)
    documents_embeddings = model.encode(sentences=documents, is_query=False)
    reranked_documents = rank.rerank(
        documents_ids=[["1", "2"]],
        queries_embeddings=queries_embeddings,
        documents_embeddings=documents_embeddings,
    )
    assert math.isclose(
        reranked_documents[0][0]["score"], 30.01, rel_tol=0.01, abs_tol=0.01
    )
    assert math.isclose(
        reranked_documents[0][1]["score"], 26.98, rel_tol=0.01, abs_tol=0.01
    )


def test_pool_embeddings_hierarchical_passes_points_to_ward(monkeypatch) -> None:
    """Ward linkage should receive Euclidean observations, not cosine distances."""
    captured = {}

    def fake_linkage(points, method):
        captured["points"] = points
        captured["method"] = method
        return np.zeros((4, 4), dtype=np.float64)

    def fake_fcluster(linkage_matrix, t, criterion):
        captured["threshold"] = t
        captured["criterion"] = criterion
        return np.array([1, 1, 2, 2, 2])

    monkeypatch.setattr(colbert.hierarchy, "linkage", fake_linkage)
    monkeypatch.setattr(colbert.hierarchy, "fcluster", fake_fcluster)

    model = object.__new__(models.ColBERT)
    embeddings = torch.arange(18, dtype=torch.float32).reshape(6, 3)

    pooled = model.pool_embeddings_hierarchical(
        documents_embeddings=[embeddings],
        pool_factor=2,
        protected_tokens=1,
    )

    assert captured["method"] == "ward"
    assert captured["points"].shape == (5, 3)
    assert captured["threshold"] == 2
    assert captured["criterion"] == "maxclust"
    assert pooled[0].shape == (3, 3)
    torch.testing.assert_close(pooled[0][0], embeddings[0])
