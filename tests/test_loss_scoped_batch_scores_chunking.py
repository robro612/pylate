from __future__ import annotations

import pytest
import torch
from torch import nn

from pylate.losses.cached_contrastive import CachedContrastive
from pylate.losses.contrastive import Contrastive
from pylate.losses.distillation import Distillation
from pylate.scores import ScopedBatchScores


def _ensure_dtype_supported(dtype: torch.dtype) -> None:
    try:
        x = torch.randn(4, 4, dtype=dtype)
        y = torch.randn(4, 4, dtype=dtype)
        _ = torch.nn.functional.normalize(x, p=2, dim=-1) @ y.T
    except Exception as exc:  # pragma: no cover - backend dependent
        pytest.skip(f"dtype {dtype} unsupported on this backend: {exc}")


class _DummyColBERT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.skiplist: list[int] = []
        self.do_query_expansion = False

    def forward(self, sentence_feature: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"token_embeddings": sentence_feature["token_embeddings"]}


def _make_feature(
    batch: int, seq_len: int, hidden: int, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    return {
        "token_embeddings": torch.nn.functional.normalize(
            torch.randn(batch, seq_len, hidden, dtype=dtype), p=2, dim=-1
        ),
        "input_ids": torch.randint(1, 100, (batch, seq_len)),
        "attention_mask": torch.ones(batch, seq_len, dtype=torch.bool),
    }


def _make_contrastive_sentence_features(
    batch: int = 4,
    groups: int = 3,
    q_len: int = 5,
    d_len: int = 7,
    hidden: int = 8,
    dtype: torch.dtype = torch.float32,
) -> list[dict[str, torch.Tensor]]:
    features = [_make_feature(batch, q_len, hidden, dtype=dtype)]
    for _ in range(groups - 1):
        features.append(_make_feature(batch, d_len, hidden, dtype=dtype))
    return features


def _make_kd_sentence_features(
    batch: int = 4,
    nway: int = 3,
    q_len: int = 5,
    d_len: int = 7,
    hidden: int = 8,
    dtype: torch.dtype = torch.float32,
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    query = _make_feature(batch, q_len, hidden, dtype=dtype)
    docs = _make_feature(batch * nway, d_len, hidden, dtype=dtype)
    labels = torch.randn(batch, nway, dtype=dtype)
    return [query, docs], labels


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_contrastive_colbert_scoped_chunking_consistent(dtype: torch.dtype) -> None:
    _ensure_dtype_supported(dtype)
    torch.manual_seed(0)
    model = _DummyColBERT()
    sentence_features = _make_contrastive_sentence_features(dtype=dtype)

    full_loss = Contrastive(
        model=model,
        score_metric=ScopedBatchScores(mode="colbert"),
    )
    chunked_loss = Contrastive(
        model=model,
        score_metric=ScopedBatchScores(
            mode="colbert",
            query_batch_chunk=2,
            doc_batch_chunk=2,
            doc_nway_chunk=2,
        ),
    )

    full_val = full_loss(sentence_features)
    chunked_val = chunked_loss(sentence_features)
    assert torch.isfinite(full_val)
    assert torch.isfinite(chunked_val)
    torch.testing.assert_close(
        chunked_val.detach().float(),
        full_val.detach().float(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_contrastive_xtr_scoped_query_chunking_consistent(dtype: torch.dtype) -> None:
    _ensure_dtype_supported(dtype)
    torch.manual_seed(1)
    model = _DummyColBERT()
    sentence_features = _make_contrastive_sentence_features(hidden=10, dtype=dtype)

    full_loss = Contrastive(
        model=model,
        score_metric=ScopedBatchScores(
            mode="xtr",
            xtr_k=8,
            xtr_topk_cast_half_for_fp32=True,
        ),
    )
    chunked_loss = Contrastive(
        model=model,
        score_metric=ScopedBatchScores(
            mode="xtr",
            query_batch_chunk=2,
            xtr_k=8,
            xtr_topk_cast_half_for_fp32=True,
        ),
    )

    full_val = full_loss(sentence_features)
    chunked_val = chunked_loss(sentence_features)
    assert torch.isfinite(full_val)
    assert torch.isfinite(chunked_val)
    torch.testing.assert_close(
        chunked_val.detach().float(),
        full_val.detach().float(),
        atol=1e-3 if dtype == torch.bfloat16 else 1e-5,
        rtol=1e-3 if dtype == torch.bfloat16 else 1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cached_contrastive_colbert_scoped_chunking_consistent(dtype: torch.dtype) -> None:
    _ensure_dtype_supported(dtype)
    torch.manual_seed(2)
    model = _DummyColBERT()
    sentence_features = _make_contrastive_sentence_features(batch=5, dtype=dtype)

    full_loss = CachedContrastive(
        model=model,
        score_metric=ScopedBatchScores(mode="colbert"),
        mini_batch_size=2,
        score_mini_batch_size=2,
    )
    chunked_loss = CachedContrastive(
        model=model,
        score_metric=ScopedBatchScores(
            mode="colbert",
            query_batch_chunk=2,
            doc_batch_chunk=2,
            doc_nway_chunk=2,
        ),
        mini_batch_size=2,
        score_mini_batch_size=2,
    )

    with torch.no_grad():
        full_val = full_loss(sentence_features)
        chunked_val = chunked_loss(sentence_features)
    assert torch.isfinite(full_val)
    assert torch.isfinite(chunked_val)
    torch.testing.assert_close(
        chunked_val.detach().float(),
        full_val.detach().float(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cached_contrastive_xtr_scoped_query_chunking_consistent(
    dtype: torch.dtype,
) -> None:
    _ensure_dtype_supported(dtype)
    torch.manual_seed(3)
    model = _DummyColBERT()
    sentence_features = _make_contrastive_sentence_features(
        batch=5, hidden=10, dtype=dtype
    )

    full_loss = CachedContrastive(
        model=model,
        score_metric=ScopedBatchScores(
            mode="xtr",
            xtr_k=8,
            xtr_topk_cast_half_for_fp32=True,
        ),
        mini_batch_size=2,
        score_mini_batch_size=2,
    )
    chunked_loss = CachedContrastive(
        model=model,
        score_metric=ScopedBatchScores(
            mode="xtr",
            query_batch_chunk=2,
            xtr_k=8,
            xtr_topk_cast_half_for_fp32=True,
        ),
        mini_batch_size=2,
        score_mini_batch_size=2,
    )

    with torch.no_grad():
        full_val = full_loss(sentence_features)
        chunked_val = chunked_loss(sentence_features)
    assert torch.isfinite(full_val)
    assert torch.isfinite(chunked_val)
    torch.testing.assert_close(
        chunked_val.detach().float(),
        full_val.detach().float(),
        atol=1e-3 if dtype == torch.bfloat16 else 1e-5,
        rtol=1e-3 if dtype == torch.bfloat16 else 1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_distillation_colbert_scoped_chunking_consistent(dtype: torch.dtype) -> None:
    _ensure_dtype_supported(dtype)
    torch.manual_seed(4)
    model = _DummyColBERT()
    sentence_features, labels = _make_kd_sentence_features(dtype=dtype)

    full_loss = Distillation(
        model=model,
        score_metric=ScopedBatchScores(mode="colbert"),
        normalize_scores=False,
    )
    chunked_loss = Distillation(
        model=model,
        score_metric=ScopedBatchScores(
            mode="colbert",
            query_batch_chunk=2,
            doc_batch_chunk=2,
            doc_nway_chunk=2,
        ),
        normalize_scores=False,
    )

    full_val = full_loss(sentence_features, labels)
    chunked_val = chunked_loss(sentence_features, labels)
    assert torch.isfinite(full_val)
    assert torch.isfinite(chunked_val)
    torch.testing.assert_close(
        chunked_val.detach().float(),
        full_val.detach().float(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_distillation_xtr_scoped_query_chunking_consistent(dtype: torch.dtype) -> None:
    _ensure_dtype_supported(dtype)
    torch.manual_seed(5)
    model = _DummyColBERT()
    sentence_features, labels = _make_kd_sentence_features(hidden=10, dtype=dtype)

    full_loss = Distillation(
        model=model,
        score_metric=ScopedBatchScores(
            mode="xtr",
            xtr_k=8,
            xtr_topk_cast_half_for_fp32=True,
        ),
        normalize_scores=False,
    )
    chunked_loss = Distillation(
        model=model,
        score_metric=ScopedBatchScores(
            mode="xtr",
            query_batch_chunk=2,
            xtr_k=8,
            xtr_topk_cast_half_for_fp32=True,
        ),
        normalize_scores=False,
    )

    full_val = full_loss(sentence_features, labels)
    chunked_val = chunked_loss(sentence_features, labels)
    assert torch.isfinite(full_val)
    assert torch.isfinite(chunked_val)
    torch.testing.assert_close(
        chunked_val.detach().float(),
        full_val.detach().float(),
        atol=1e-3 if dtype == torch.bfloat16 else 1e-5,
        rtol=1e-3 if dtype == torch.bfloat16 else 1e-5,
    )
