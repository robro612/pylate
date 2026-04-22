from __future__ import annotations

import pytest
import torch

from pylate.scores import ScopedBatchScores, XTRScores, colbert_scores


@pytest.mark.parametrize(
    ("query_batch_chunk", "doc_batch_chunk", "doc_nway_chunk"),
    [
        (None, None, None),
        (1, None, None),
        (2, None, None),
        (2, 1, None),
        (2, 3, 1),
        (2, 3, 2),
        (3, 2, 2),
        (8, 10, 10),  # chunk > dimension should behave like full chunk
    ],
)
def test_colbert_chunking_matches_full(
    query_batch_chunk: int | None,
    doc_batch_chunk: int | None,
    doc_nway_chunk: int | None,
) -> None:
    torch.manual_seed(13)
    B, N, Q, D, H = 5, 3, 4, 6, 8

    queries = torch.randn(B, Q, H)
    documents = torch.randn(B, N, D, H)
    queries_mask = (torch.rand(B, Q) > 0.2).float()
    documents_mask = (torch.rand(B, N, D) > 0.15).float()

    baseline = ScopedBatchScores(mode="colbert")(
        queries_embeddings=queries,
        documents_embeddings=documents,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
    )
    chunked = ScopedBatchScores(
        mode="colbert",
        query_batch_chunk=query_batch_chunk,
        doc_batch_chunk=doc_batch_chunk,
        doc_nway_chunk=doc_nway_chunk,
    )(
        queries_embeddings=queries,
        documents_embeddings=documents,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
    )

    assert baseline.shape == (B, B * N)
    torch.testing.assert_close(chunked, baseline, atol=1e-5, rtol=1e-5)


def test_colbert_matches_existing_colbert_scores() -> None:
    torch.manual_seed(23)
    B, N, Q, D, H = 4, 3, 6, 7, 9

    queries = torch.randn(B, Q, H)
    documents = torch.randn(B, N, D, H)
    queries_mask = (torch.rand(B, Q) > 0.2).float()
    documents_mask = (torch.rand(B, N, D) > 0.15).float()

    docs_flat = documents.view(B * N, D, H)
    docs_mask_flat = documents_mask.view(B * N, D)

    original = colbert_scores(
        queries_embeddings=queries,
        documents_embeddings=docs_flat,
        queries_mask=queries_mask,
        documents_mask=docs_mask_flat,
    )
    new = ScopedBatchScores(mode="colbert")(
        queries_embeddings=queries,
        documents_embeddings=documents,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
    )

    assert original.shape == (B, B * N)
    torch.testing.assert_close(new, original, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("query_batch_chunk", [None, 1, 2, 3, 8, 16])
def test_xtr_query_chunking_matches_full(query_batch_chunk: int | None) -> None:
    torch.manual_seed(7)
    B, N, Q, D, H = 6, 2, 5, 7, 9

    queries = torch.randn(B, Q, H)
    documents = torch.randn(B, N, D, H)
    queries_mask = (torch.rand(B, Q) > 0.2).float()
    documents_mask = (torch.rand(B, N, D) > 0.15).float()

    baseline = ScopedBatchScores(mode="xtr", xtr_k=8)(
        queries_embeddings=queries,
        documents_embeddings=documents,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
    )
    chunked = ScopedBatchScores(
        mode="xtr", xtr_k=8, query_batch_chunk=query_batch_chunk
    )(
        queries_embeddings=queries,
        documents_embeddings=documents,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
    )

    assert baseline.shape == (B, B * N)
    torch.testing.assert_close(chunked, baseline, atol=1e-5, rtol=1e-5)


def test_xtr_matches_existing_xtrscores() -> None:
    torch.manual_seed(17)
    B, N, Q, D, H = 5, 3, 4, 8, 10
    k = 12

    queries = torch.randn(B, Q, H)
    documents = torch.randn(B, N, D, H)
    queries_mask = (torch.rand(B, Q) > 0.25).float()
    documents_mask = (torch.rand(B, N, D) > 0.2).float()

    original = XTRScores(k=k)(
        queries_embeddings=queries,
        documents_embeddings=documents,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
    )
    new = ScopedBatchScores(mode="xtr", xtr_k=k)(
        queries_embeddings=queries,
        documents_embeddings=documents,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
    )

    assert isinstance(original, torch.Tensor)
    assert isinstance(new, torch.Tensor)
    assert original.shape == (B, B * N)
    torch.testing.assert_close(new, original, atol=1e-5, rtol=1e-5)


def test_xtr_rejects_doc_chunking() -> None:
    with pytest.raises(ValueError, match="doc_batch_chunk"):
        ScopedBatchScores(mode="xtr", doc_batch_chunk=2)
    with pytest.raises(ValueError, match="doc_nway_chunk"):
        ScopedBatchScores(mode="xtr", doc_nway_chunk=2)
    with pytest.raises(ValueError, match="doc_batch_chunk"):
        ScopedBatchScores(mode="xtr", doc_batch_chunk=2, doc_nway_chunk=2)


def test_scope_constraints() -> None:
    with pytest.raises(ValueError, match="return_scope='global'"):
        ScopedBatchScores(
            mode="colbert",
            scoring_scope="local",
            return_scope="global",
        )

    with pytest.raises(ValueError, match="XTR requires scoring_scope='global'"):
        ScopedBatchScores(mode="xtr", scoring_scope="local")

    with pytest.raises(ValueError, match="doc_batch_chunk is not supported"):
        ScopedBatchScores(
            mode="colbert",
            scoring_scope="local",
            return_scope="local",
            doc_batch_chunk=2,
        )


@pytest.mark.parametrize("mode", ["colbert", "xtr"])
def test_return_scope_local_shape(mode: str) -> None:
    torch.manual_seed(31)
    B, N, Q, D, H = 4, 3, 5, 6, 8
    queries = torch.randn(B, Q, H)
    documents = torch.randn(B, N, D, H)
    q_mask = (torch.rand(B, Q) > 0.2).float()
    d_mask = (torch.rand(B, N, D) > 0.2).float()

    scorer = ScopedBatchScores(
        mode=mode,
        scoring_scope="global",
        return_scope="local",
        query_batch_chunk=2,
        xtr_k=8,
    )
    scores = scorer(
        queries_embeddings=queries,
        documents_embeddings=documents,
        queries_mask=q_mask,
        documents_mask=d_mask,
    )
    assert scores.shape == (B, N)


def test_colbert_local_scoring_matches_global_local_slice() -> None:
    torch.manual_seed(37)
    B, N, Q, D, H = 5, 2, 4, 6, 8
    queries = torch.randn(B, Q, H)
    documents = torch.randn(B, N, D, H)
    q_mask = (torch.rand(B, Q) > 0.2).float()
    d_mask = (torch.rand(B, N, D) > 0.2).float()

    global_local = ScopedBatchScores(
        mode="colbert",
        scoring_scope="global",
        return_scope="local",
        query_batch_chunk=2,
        doc_batch_chunk=3,
        doc_nway_chunk=1,
    )(
        queries_embeddings=queries,
        documents_embeddings=documents,
        queries_mask=q_mask,
        documents_mask=d_mask,
    )
    local_local = ScopedBatchScores(
        mode="colbert",
        scoring_scope="local",
        return_scope="local",
        query_batch_chunk=2,
        doc_nway_chunk=1,
    )(
        queries_embeddings=queries,
        documents_embeddings=documents,
        queries_mask=q_mask,
        documents_mask=d_mask,
    )
    torch.testing.assert_close(local_local, global_local, atol=1e-5, rtol=1e-5)
