"""Tests for quantization-aware training via the straight-through estimator."""

from __future__ import annotations

import os
import shutil

import pytest
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers

from pylate import losses, models, utils
from pylate.models import (
    BinaryQuantizer,
    IdentityQuantizer,
    ScalarQuantizer,
    StraightThroughEstimator,
    straight_through,
)


def test_straight_through_identity_gradient() -> None:
    """Forward value is the quantized one; gradient w.r.t. the input is identity."""
    x = torch.randn(4, 8, requires_grad=True)
    quantized = torch.round(x)  # non-differentiable on its own
    y = straight_through(x, quantized)

    assert torch.equal(y, quantized)

    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_scalar_quantizer_grid() -> None:
    """ScalarQuantizer output lies on a 2**n_bits grid and is shape-preserving."""
    q = ScalarQuantizer(n_bits=2)  # 4 levels
    x = torch.randn(5, 64)
    out = q(x)
    assert out.shape == x.shape
    # Per-vector affine quant with 4 levels => at most 4 distinct values per row.
    for row in out:
        assert row.unique().numel() <= 4


def test_binary_quantizer_is_binary() -> None:
    out = BinaryQuantizer(scale=2.0)(torch.randn(3, 16))
    assert torch.all(out.abs() == 2.0)


def test_asymmetric_routing() -> None:
    """Queries and documents are routed to different transforms via is_query."""
    ste = StraightThroughEstimator(
        query_transform=ScalarQuantizer(n_bits=8),
        document_transform=BinaryQuantizer(),
    )
    assert ste.asymmetric

    query_out = ste({"token_embeddings": torch.randn(4, 32), "is_query": True})[
        "token_embeddings"
    ]
    document_out = ste({"token_embeddings": torch.randn(4, 32), "is_query": False})[
        "token_embeddings"
    ]

    # Documents are binarized, queries are int8 (many distinct levels).
    assert torch.all(document_out.abs() == document_out.abs()[0, 0])
    assert document_out.unique().numel() <= 2
    assert query_out.unique().numel() > 2

    # A tensor is_query flag (as produced after collation) also routes correctly.
    document_out_tensor_flag = ste(
        {"token_embeddings": torch.randn(4, 32), "is_query": torch.tensor(False)}
    )["token_embeddings"]
    assert document_out_tensor_flag.unique().numel() <= 2


def test_symmetric_default_and_grad_flow() -> None:
    """A symmetric STE applies one transform and lets gradients reach the input."""
    ste = StraightThroughEstimator(transform=ScalarQuantizer(n_bits=4))
    assert not ste.asymmetric

    x = torch.randn(3, 16, requires_grad=True)
    out = ste({"token_embeddings": x})["token_embeddings"]
    out.pow(2).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_config_roundtrip(tmp_path) -> None:
    """Saving then loading the module preserves the asymmetric configuration."""
    ste = StraightThroughEstimator(
        query_transform=ScalarQuantizer(n_bits=8, symmetric=True),
        document_transform=BinaryQuantizer(scale=0.5),
        pre_normalize=False,
        default_is_query=True,
    )
    ste.save(str(tmp_path))
    loaded = StraightThroughEstimator.load(str(tmp_path))

    assert loaded.asymmetric
    assert loaded.pre_normalize is False
    assert loaded.default_is_query is True

    config = loaded.get_config_dict()
    assert config["query"]["quantizer_type"] == "ScalarQuantizer"
    assert config["query"]["quantizer_config"]["symmetric"] is True
    assert config["query"]["quantizer_config"]["n_bits"] == 8
    assert config["document"]["quantizer_type"] == "BinaryQuantizer"
    assert config["document"]["quantizer_config"]["scale"] == 0.5


def test_non_serializable_transform_falls_back_to_identity(tmp_path) -> None:
    """A plain-callable transform trains but loads back as a no-op identity."""
    ste = StraightThroughEstimator(transform=lambda t: torch.round(t * 4) / 4)
    # The callable still works on the forward pass.
    out = ste({"token_embeddings": torch.randn(2, 8)})["token_embeddings"]
    assert out.shape == (2, 8)

    ste.save(str(tmp_path))
    loaded = StraightThroughEstimator.load(str(tmp_path))
    assert isinstance(loaded._transform_for(is_query=False), IdentityQuantizer)


@pytest.mark.skipif(torch.backends.mps.is_available(), reason="MPS is not supported")
def test_asymmetric_quantization_aware_training() -> None:
    """End-to-end: train a ColBERT with int8 queries / binary documents, then
    save and reload the model with the STE module intact and active."""
    if os.path.exists(path="tests/qat"):
        shutil.rmtree("tests/qat")

    model = models.ColBERT(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
    # Permanent, asymmetric quantization module appended after the Dense layer.
    model.append(
        StraightThroughEstimator(
            query_transform=ScalarQuantizer(n_bits=8),
            document_transform=BinaryQuantizer(),
        )
    )
    assert isinstance(model[-1], StraightThroughEstimator)

    dataset = load_dataset("lightonai/lighton-ms-marco-mini", "triplet", split="train")
    train_dataset = dataset.train_test_split(test_size=0.5)["train"]

    train_loss = losses.Contrastive(model=model)

    args = SentenceTransformerTrainingArguments(
        output_dir="tests/qat",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        fp16=False,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        learning_rate=3e-6,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    trainer.train()
    assert trainer.state.log_history  # training produced at least one loss value

    model.save_pretrained("tests/qat/final")

    # Reload: the STE module must round-trip with its asymmetric configuration.
    reloaded = models.ColBERT(model_name_or_path="tests/qat/final")
    assert isinstance(reloaded[-1], StraightThroughEstimator)
    assert reloaded[-1].asymmetric

    # The module is active at inference: documents come out binary, queries int8.
    query_emb = reloaded.encode(
        ["what is the capital of france?"],
        is_query=True,
        convert_to_tensor=True,
    )[0]
    document_emb = reloaded.encode(
        ["paris is the capital of france."],
        is_query=False,
        convert_to_tensor=True,
    )[0]

    # encode() re-normalizes, so binary documents have a single magnitude per row.
    assert document_emb.unique().numel() <= 2
    assert query_emb.unique().numel() > 2

    if os.path.exists(path="tests/qat"):
        shutil.rmtree("tests/qat")
