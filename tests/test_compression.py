"""Tests for compression-aware training: SHBQ, token poolers, and the loss."""

from __future__ import annotations

from unittest import mock

import pytest
import torch

import numpy as np

from pylate import losses, models
from pylate.models import (
    BinaryQuantizer,
    CastQuantizer,
    Compressor,
    KMeansPooler,
    KMeansPoolingConfig,
    SHBQQuantizer,
    WardPooler,
    WardPoolingConfig,
    build_compressor,
    build_quantizer,
)


# --------------------------------------------------------------------------- #
# SHBQ quantizer
# --------------------------------------------------------------------------- #
def test_shbq_is_rotation_around_inner_quantizer() -> None:
    """SHBQ == rotate, run the inner quantizer, rotate back. Default inner is binary."""
    torch.manual_seed(0)
    q = SHBQQuantizer(dim=64)  # default inner: BinaryQuantizer
    assert isinstance(q.inner, BinaryQuantizer)
    x = torch.randn(5, 64)

    h, d = q.H, q.d
    rotated = (x * d) @ h
    expected = (q.inner(rotated) @ h) * d

    torch.testing.assert_close(q(x), expected)
    # The rotated representation is strictly binary (two levels).
    assert torch.sign(rotated).unique().numel() == 2


def test_shbq_preserves_dot_products_across_query_and_document() -> None:
    """Two SHBQ quantizers with the same (dim, seed) score like the rotated binary
    dot product, because the Hadamard transform is an orthonormal rotation."""
    torch.manual_seed(0)
    query_q = SHBQQuantizer(dim=64, seed=7)
    doc_q = SHBQQuantizer(dim=64, seed=7)

    a, b = torch.randn(3, 64), torch.randn(4, 64)
    dequantized_scores = query_q(a) @ doc_q(b).T

    # Reference: binary codes compared directly in the rotated space.
    rotated_a = torch.sign((a * query_q.d) @ query_q.H)
    rotated_b = torch.sign((b * doc_q.d) @ doc_q.H)
    rotated_scores = rotated_a @ rotated_b.T

    torch.testing.assert_close(dequantized_scores, rotated_scores, atol=1e-4, rtol=1e-4)


def test_shbq_int8_inner_quantizes_in_rotated_space() -> None:
    """An int8 inner quantizes the rotated vector to many levels (the int8 query path)."""
    torch.manual_seed(0)
    query = SHBQQuantizer(inner=CastQuantizer("int8"), dim=64)
    x = torch.randn(4, 64)

    rotated = (x * query.d) @ query.H
    levels = (query.inner(rotated) / rotated.abs().amax(dim=-1, keepdim=True) * 127)
    assert levels.round().unique().numel() > 2
    assert query(x).shape == x.shape


def test_shbq_requires_power_of_two_dim() -> None:
    with pytest.raises(ValueError, match="power of two"):
        SHBQQuantizer(dim=48)
    lazy = SHBQQuantizer()  # inferred lazily
    with pytest.raises(ValueError, match="power of two"):
        lazy(torch.randn(2, 48))


def test_shbq_straight_through_gradient_is_identity() -> None:
    """Wrapped in a straight-through estimator, the gradient is identity."""
    torch.manual_seed(0)
    q = SHBQQuantizer(dim=32)
    x = torch.randn(3, 32, requires_grad=True)
    y = models.straight_through(x, q(x))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


# --------------------------------------------------------------------------- #
# CastQuantizer
# --------------------------------------------------------------------------- #
def test_cast_quantizer_float_formats() -> None:
    """Float dtypes do a true IEEE cast; fp32 is a no-op."""
    torch.manual_seed(0)
    x = torch.randn(3, 16)
    torch.testing.assert_close(CastQuantizer("float32")(x), x)
    torch.testing.assert_close(CastQuantizer("fp16")(x), x.half().float())
    torch.testing.assert_close(CastQuantizer("bf16")(x), x.bfloat16().float())


def test_cast_quantizer_int8_is_uniform_grid() -> None:
    """int8 lands on a 127-level symmetric grid and stays shape-preserving."""
    torch.manual_seed(0)
    x = torch.randn(2, 32)
    q = CastQuantizer("int8")(x)
    assert q.shape == x.shape
    scale = x.abs().amax(dim=-1, keepdim=True) / 127
    assert (q / scale).round().abs().max() <= 127


def test_cast_quantizer_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="Unknown dtype"):
        CastQuantizer("int3")


# --------------------------------------------------------------------------- #
# Poolers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pooler_factory", ["kmeans", "ward"])
def test_pooler_reduces_token_count_and_keeps_protected(pooler_factory: str) -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(2, 24, 16)
    mask = torch.ones(2, 24, dtype=torch.bool)

    if pooler_factory == "kmeans":
        pooler = KMeansPooler(KMeansPoolingConfig(pool_factor=4, protected_tokens=2))
    else:
        pooler = WardPooler(WardPoolingConfig(pool_factor=4, protected_tokens=2))

    pooled, pooled_mask = pooler(embeddings, mask)

    assert pooled.shape[1] < embeddings.shape[1]
    assert pooled_mask.shape == pooled.shape[:2]
    # Protected tokens bypass pooling and are kept verbatim at the front.
    torch.testing.assert_close(pooled[:, :2], embeddings[:, :2])


def test_pooler_respects_mask_and_is_differentiable() -> None:
    """Only un-masked tokens are pooled, and gradients reach the inputs."""
    torch.manual_seed(0)
    embeddings = torch.randn(2, 24, 16, requires_grad=True)
    mask = torch.ones(2, 24, dtype=torch.bool)
    mask[0, 18:] = False  # sample 0 has only 18 valid tokens

    pooler = KMeansPooler(KMeansPoolingConfig(pool_factor=3, protected_tokens=1))
    pooled, _ = pooler(embeddings, mask)
    pooled.pow(2).sum().backward()

    assert embeddings.grad is not None
    # Masked-out positions of sample 0 receive no gradient.
    assert embeddings.grad[0, 18:].abs().sum() == 0
    assert embeddings.grad[0, :18].abs().sum() > 0


def test_ward_threshold_is_adaptive_per_document() -> None:
    """A cosine similarity_threshold yields a data-dependent cluster count."""
    torch.manual_seed(0)
    embeddings = torch.randn(3, 30, 16)
    mask = torch.ones(3, 30, dtype=torch.bool)

    pooler = WardPooler(
        WardPoolingConfig(similarity_threshold=0.5, protected_tokens=1)
    )
    pooled, pooled_mask = pooler(embeddings, mask)
    lengths = pooled_mask.sum(dim=1)

    assert pooled.shape[1] <= embeddings.shape[1]
    assert lengths.max() > 0


def test_pooling_config_validation() -> None:
    with pytest.raises(ValueError, match="pool_factor"):
        KMeansPoolingConfig(pool_factor=0)
    with pytest.raises(ValueError, match="protected_tokens"):
        WardPoolingConfig(protected_tokens=-1)
    with pytest.raises(ValueError, match="similarity_threshold"):
        WardPoolingConfig(similarity_threshold=2.0)
    with pytest.raises(ValueError, match="normalize_input"):
        WardPoolingConfig(similarity_threshold=0.5, normalize_input=False)


# --------------------------------------------------------------------------- #
# Compressor pipeline
# --------------------------------------------------------------------------- #
def test_compressor_identity_is_normalized_passthrough() -> None:
    compressor = Compressor()
    embeddings = torch.randn(2, 10, 16)
    mask = torch.ones(2, 10, dtype=torch.bool)
    out, out_mask = compressor(embeddings, mask)

    assert compressor.is_identity
    assert out.shape == embeddings.shape
    torch.testing.assert_close(out, torch.nn.functional.normalize(embeddings, dim=-1))
    torch.testing.assert_close(out_mask, mask)


def test_compressor_composes_pool_then_quantize() -> None:
    """Pooling runs first (token count drops); quantization keeps that shape."""
    torch.manual_seed(0)
    compressor = Compressor(
        quantizer=SHBQQuantizer(dim=16),
        pooler=KMeansPooler(KMeansPoolingConfig(pool_factor=4, protected_tokens=1)),
    )
    embeddings = torch.randn(2, 24, 16)
    mask = torch.ones(2, 24, dtype=torch.bool)
    out, out_mask = compressor(embeddings, mask)

    assert out.shape[1] < embeddings.shape[1]
    assert out.shape[:2] == out_mask.shape


def test_compressor_from_config_roundtrip() -> None:
    spec = {
        "quantizer": {"type": "SHBQQuantizer", "dim": 16, "inner": {"type": "BinaryQuantizer"}},
        "pooler": {"name": "ward", "num_clusters": 4, "protected_tokens": 1},
    }
    compressor = build_compressor(spec)
    assert isinstance(compressor.quantizer, SHBQQuantizer)
    assert isinstance(compressor.pooler, WardPooler)
    assert not compressor.is_identity

    # get_config_dict <-> from_config round-trips, so a compressor saved after
    # training rebuilds identically at index time.
    rebuilt = build_compressor(compressor.get_config_dict())
    assert rebuilt.get_config_dict() == compressor.get_config_dict()
    assert build_compressor(compressor) is compressor  # pass-through


@pytest.mark.parametrize(
    "compressor_factory",
    [
        lambda: Compressor(quantizer=CastQuantizer("int8")),
        lambda: Compressor(quantizer=SHBQQuantizer(dim=16)),
        lambda: Compressor(
            pooler=KMeansPooler(KMeansPoolingConfig(pool_factor=4, protected_tokens=1))
        ),
        lambda: Compressor(
            quantizer=CastQuantizer("int8"),
            pooler=KMeansPooler(KMeansPoolingConfig(num_clusters=5, protected_tokens=1)),
        ),
    ],
)
def test_compressor_train_and_inference_paths_agree(compressor_factory) -> None:
    """The same Compressor yields identical values in training (batched) and at
    inference (per-document encode output)."""
    torch.manual_seed(0)
    compressor = compressor_factory()
    doc = torch.randn(24, 16)

    # Training path: a batch of one, all tokens valid.
    batched, mask = compressor(doc.unsqueeze(0), torch.ones(1, 24, dtype=torch.bool))
    train_out = batched[0][mask[0]]

    # Inference path: the per-document tensor encode() returns.
    infer_out = compressor.transform_one(doc)

    assert train_out.shape == infer_out.shape
    torch.testing.assert_close(train_out, infer_out)


def test_compressor_transform_accepts_numpy_and_lists() -> None:
    compressor = Compressor(quantizer=CastQuantizer("int8"))
    docs = [np.random.randn(12, 16).astype("float32"), np.random.randn(7, 16).astype("float32")]
    out = compressor.transform(docs)

    assert len(out) == 2
    assert all(isinstance(o, np.ndarray) for o in out)
    assert out[0].shape == (12, 16)


# --------------------------------------------------------------------------- #
# CompressionAwareLoss (requires a small ColBERT model)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def model() -> models.ColBERT:
    return models.ColBERT(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    )


def _features(model: models.ColBERT):
    anchor = model.preprocess(["fruits are healthy."], is_query=True)
    positive = model.preprocess(["fruits are good for health."], is_query=False)
    negative = model.preprocess(["rocks are hard and grey."], is_query=False)
    return [anchor, positive, negative]


def test_compression_aware_lambda_one_matches_base_loss(model) -> None:
    """lam=1 with identity compressors recovers the plain base loss exactly."""
    features = _features(model)
    base = losses.Contrastive(model=model)
    wrapped = losses.CompressionAwareLoss(loss=base, lam=1.0)

    torch.testing.assert_close(wrapped(features), base(features))


def test_compression_aware_single_model_pass(model) -> None:
    """The model is embedded once; the base loss scores raw and compressed twice."""
    base = losses.Contrastive(model=model)
    wrapped = losses.CompressionAwareLoss(
        loss=base,
        document_compressor=Compressor(quantizer=SHBQQuantizer()),
        lam=0.5,
    )
    features = _features(model)

    with (
        mock.patch.object(base, "embed", wraps=base.embed) as embed_spy,
        mock.patch.object(
            base, "loss_from_embeddings", wraps=base.loss_from_embeddings
        ) as score_spy,
    ):
        wrapped(features)

    assert embed_spy.call_count == 1
    assert score_spy.call_count == 2


def test_compression_aware_gradient_reaches_model(model) -> None:
    base = losses.Contrastive(model=model)
    wrapped = losses.CompressionAwareLoss(
        loss=base,
        query_compressor=Compressor(quantizer=CastQuantizer("int8")),
        document_compressor=Compressor(
            quantizer=SHBQQuantizer(),
            pooler=WardPooler(WardPoolingConfig(num_clusters=4)),
        ),
        lam=0.5,
    )
    model.zero_grad()
    wrapped(_features(model)).backward()

    grads = [
        p.grad.abs().sum().item()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert grads and sum(grads) > 0


def test_compression_aware_lambda_blends_the_two_losses(model) -> None:
    """The mixed loss is the lam-weighted average of the full and compressed terms."""
    features = _features(model)
    base = losses.Contrastive(model=model)
    doc_compressor = Compressor(quantizer=SHBQQuantizer())

    full = losses.CompressionAwareLoss(loss=base, lam=1.0)(features)
    compressed = losses.CompressionAwareLoss(
        loss=base, document_compressor=doc_compressor, lam=0.0
    )(features)
    mixed = losses.CompressionAwareLoss(
        loss=base, document_compressor=doc_compressor, lam=0.25
    )(features)

    expected = 0.25 * full + 0.75 * compressed
    torch.testing.assert_close(mixed, expected, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Library factories: pylate.data RLHN + build_compression_aware_loss
# --------------------------------------------------------------------------- #
def test_rlhn_to_contrastive_flattens_to_query_first_columns() -> None:
    """RLHN rows flatten to query/positive/negative_1..k with the query first."""
    from datasets import Dataset

    from pylate import data

    def passage(text, title=""):
        return {"docid": text[:4], "text": text, "title": title}

    raw = Dataset.from_list(
        [
            {
                "query": "what are intranets",
                "positive_passages": [passage("an intranet is a private network")],
                "negative_passages": [passage(f"unrelated text {i}") for i in range(5)],
            },
            {
                "query": "capital of france",
                "positive_passages": [passage("paris is the capital", title="France")],
                "negative_passages": [passage("a hard negative")],  # only one -> resampled
            },
        ]
    )
    flat = data.rlhn_to_contrastive(raw, num_negatives=3, seed=0)

    assert flat.column_names == ["query", "positive", "negative_1", "negative_2", "negative_3"]
    assert flat[0]["query"] == "what are intranets"
    # title is prepended when present.
    assert flat[1]["positive"].startswith("France. paris")
    # exactly num_negatives columns even when fewer hard negatives were available.
    assert all(flat[1][f"negative_{i}"] for i in (1, 2, 3))


def test_build_compression_aware_loss_from_spec(model) -> None:
    """The config-block factory builds the same loss the runner would."""
    spec = {
        "lambda": 0.5,
        "query": {"quantizer": {"type": "CastQuantizer", "dtype": "int8"}},
        "document": {
            "quantizer": {"type": "SHBQQuantizer", "inner": {"type": "BinaryQuantizer"}},
            "pooler": {"name": "ward", "num_clusters": 4},
        },
    }
    loss = losses.build_compression_aware_loss(losses.Contrastive(model=model), spec)

    assert isinstance(loss, losses.CompressionAwareLoss)
    assert loss.lam == 0.5
    assert isinstance(loss.query_compressor.quantizer, CastQuantizer)
    assert isinstance(loss.document_compressor.quantizer, SHBQQuantizer)
    assert isinstance(loss.document_compressor.pooler, WardPooler)
    assert isinstance(loss(_features(model)).item(), float)
