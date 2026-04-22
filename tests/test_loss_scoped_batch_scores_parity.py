from __future__ import annotations

import pytest
import torch
from torch import nn

from pylate.losses.cached_contrastive import CachedContrastive, CachedContrastive_New
from pylate.losses.contrastive import Contrastive, Contrastive_New
from pylate.losses.distillation import Distillation, Distillation_New
from pylate.scores import (
    ScopedBatchScores,
    XTRKDScores,
    XTRScores,
    colbert_kd_scores,
    colbert_scores,
)


def _report_and_assert_close(
    name: str,
    old_val: torch.Tensor,
    new_val: torch.Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    old_f = float(old_val.detach().cpu().item())
    new_f = float(new_val.detach().cpu().item())
    abs_diff = abs(new_f - old_f)
    rel_diff = abs_diff / max(abs(old_f), 1e-12)
    print(
        f"{name}: old={old_f:.12f} new={new_f:.12f} "
        f"abs_diff={abs_diff:.12e} rel_diff={rel_diff:.12e}"
    )
    # Compare values in fp32; low-precision paths may legitimately differ in dtype.
    torch.testing.assert_close(
        new_val.detach().float(), old_val.detach().float(), atol=atol, rtol=rtol
    )


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
def test_contrastive_colbert_scoped_parity(dtype: torch.dtype) -> None:
    _ensure_dtype_supported(dtype)
    torch.manual_seed(0)
    model = _DummyColBERT()
    sentence_features = _make_contrastive_sentence_features(dtype=dtype)

    old_loss = Contrastive(model=model, score_metric=colbert_scores)
    new_loss = Contrastive_New(
        model=model,
        score_metric=ScopedBatchScores(
            mode="colbert", scoring_scope="local", return_scope="local"
        ),
    )

    old_val = old_loss(sentence_features)
    new_val = new_loss(sentence_features)
    _report_and_assert_close(f"contrastive_colbert[{dtype}]", old_val, new_val)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_contrastive_xtr_scoped_parity(dtype: torch.dtype) -> None:
    _ensure_dtype_supported(dtype)
    if dtype == torch.float16:
        pytest.skip("Legacy XTR path overflows in fp16 due masked_fill sentinel.")
    torch.manual_seed(1)
    model = _DummyColBERT()
    sentence_features = _make_contrastive_sentence_features(hidden=10, dtype=dtype)

    old_loss = Contrastive(model=model, score_metric=XTRScores(k=8))
    new_loss = Contrastive_New(
        model=model,
        score_metric=ScopedBatchScores(
            mode="xtr",
            scoring_scope="global",
            return_scope="local",
            xtr_k=8,
            xtr_topk_cast_half_for_fp32=True,
        ),
    )

    old_val = old_loss(sentence_features)
    new_val = new_loss(sentence_features)
    if dtype == torch.bfloat16:
        _report_and_assert_close(
            f"contrastive_xtr[{dtype}]", old_val, new_val, atol=2e-2, rtol=2e-2
        )
    else:
        _report_and_assert_close(f"contrastive_xtr[{dtype}]", old_val, new_val)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cached_contrastive_colbert_scoped_parity(dtype: torch.dtype) -> None:
    _ensure_dtype_supported(dtype)
    torch.manual_seed(2)
    model = _DummyColBERT()
    sentence_features = _make_contrastive_sentence_features(batch=5, dtype=dtype)

    old_loss = CachedContrastive(
        model=model,
        score_metric=colbert_scores,
        mini_batch_size=2,
        score_mini_batch_size=2,
    )
    new_loss = CachedContrastive_New(
        model=model,
        score_metric=ScopedBatchScores(
            mode="colbert", scoring_scope="local", return_scope="local"
        ),
        mini_batch_size=2,
        score_mini_batch_size=2,
    )

    with torch.no_grad():
        old_val = old_loss(sentence_features)
        new_val = new_loss(sentence_features)
    _report_and_assert_close(
        f"cached_contrastive_colbert[{dtype}]", old_val, new_val
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cached_contrastive_xtr_scoped_parity(dtype: torch.dtype) -> None:
    _ensure_dtype_supported(dtype)
    if dtype == torch.float16:
        pytest.skip("Legacy XTR path overflows in fp16 due masked_fill sentinel.")
    torch.manual_seed(3)
    model = _DummyColBERT()
    sentence_features = _make_contrastive_sentence_features(
        batch=5, hidden=10, dtype=dtype
    )

    old_loss = CachedContrastive(
        model=model,
        score_metric=XTRScores(k=8),
        mini_batch_size=2,
        score_mini_batch_size=2,
    )
    new_loss = CachedContrastive_New(
        model=model,
        score_metric=ScopedBatchScores(
            mode="xtr",
            scoring_scope="global",
            return_scope="local",
            xtr_k=8,
            xtr_topk_cast_half_for_fp32=True,
        ),
        mini_batch_size=2,
        score_mini_batch_size=2,
    )

    with torch.no_grad():
        old_val = old_loss(sentence_features)
        new_val = new_loss(sentence_features)
    if dtype == torch.bfloat16:
        _report_and_assert_close(
            f"cached_contrastive_xtr[{dtype}]",
            old_val,
            new_val,
            atol=2e-2,
            rtol=2e-2,
        )
    else:
        _report_and_assert_close(f"cached_contrastive_xtr[{dtype}]", old_val, new_val)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_distillation_colbert_scoped_parity(dtype: torch.dtype) -> None:
    _ensure_dtype_supported(dtype)
    torch.manual_seed(4)
    model = _DummyColBERT()
    sentence_features, labels = _make_kd_sentence_features(dtype=dtype)

    old_loss = Distillation(
        model=model, score_metric=colbert_kd_scores, normalize_scores=False
    )
    new_loss = Distillation_New(
        model=model,
        score_metric=ScopedBatchScores(
            mode="colbert", scoring_scope="global", return_scope="global"
        ),
        normalize_scores=False,
    )

    old_val = old_loss(sentence_features, labels)
    new_val = new_loss(sentence_features, labels)
    _report_and_assert_close(f"distillation_colbert[{dtype}]", old_val, new_val)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_distillation_xtr_scoped_parity(dtype: torch.dtype) -> None:
    _ensure_dtype_supported(dtype)
    if dtype == torch.float16:
        pytest.skip("Legacy XTR path overflows in fp16 due masked_fill sentinel.")
    torch.manual_seed(5)
    model = _DummyColBERT()
    sentence_features, labels = _make_kd_sentence_features(hidden=10, dtype=dtype)

    old_loss = Distillation(
        model=model, score_metric=XTRKDScores(k=8), normalize_scores=False
    )
    new_loss = Distillation_New(
        model=model,
        score_metric=ScopedBatchScores(
            mode="xtr",
            scoring_scope="global",
            return_scope="global",
            xtr_k=8,
            xtr_topk_cast_half_for_fp32=True,
        ),
        normalize_scores=False,
    )

    old_val = old_loss(sentence_features, labels)
    new_val = new_loss(sentence_features, labels)
    if dtype == torch.bfloat16:
        _report_and_assert_close(
            f"distillation_xtr[{dtype}]",
            old_val,
            new_val,
            atol=2e-2,
            rtol=2e-2,
        )
    else:
        _report_and_assert_close(f"distillation_xtr[{dtype}]", old_val, new_val)
