from __future__ import annotations

import pytest
import torch

from pylate import models


@pytest.mark.parametrize("variant", ["flatten", "transpose"])
def test_constbert_tokenize_and_forward(variant: str) -> None:
    model = models.ConstBERT(
        model_name_or_path="bert-base-uncased",
        constbert_variant=variant,
        constbert_seq_length=6,
        query_length=8,
        document_length=12,
    )

    documents = ["hello world", "quick brown fox"]
    features = model.tokenize(texts=documents, is_query=False)
    assert features["input_ids"].shape[1] == model.document_length

    queries = ["what is this", "another query"]
    query_features = model.tokenize(texts=queries, is_query=True)
    assert query_features["input_ids"].shape[1] == model.query_length

    features = {k: v.to(model.device) for k, v in features.items()}
    with torch.no_grad():
        out_features = model.forward(input=features, is_query=False)

    assert out_features["token_embeddings"].shape[1] == model.constbert_seq_length
    assert out_features["attention_mask"].shape[1] == model.constbert_seq_length


def test_constbert_encode_shapes() -> None:
    model = models.ConstBERT(
        model_name_or_path="bert-base-uncased",
        constbert_variant="flatten",
        constbert_seq_length=6,
        query_length=8,
        document_length=12,
    )

    documents = ["hello world", "quick brown fox"]
    embeddings = model.encode(
        sentences=documents,
        is_query=False,
        convert_to_tensor=True,
    )

    assert len(embeddings) == len(documents)
    for embedding in embeddings:
        assert embedding.shape[0] == model.constbert_seq_length


def test_constbert_save_and_load(tmp_path) -> None:
    model = models.ConstBERT(
        model_name_or_path="bert-base-uncased",
        constbert_variant="transpose",
        constbert_seq_length=5,
        query_length=7,
        document_length=11,
    )

    output_dir = tmp_path / "constbert_model"
    model.save_pretrained(str(output_dir))

    loaded = models.ConstBERT.load(str(output_dir))
    assert loaded.projection_variant == model.projection_variant
    assert loaded.constbert_seq_length == model.constbert_seq_length

    direct_loaded = models.ConstBERT(str(output_dir))
    assert direct_loaded.projection_variant == model.projection_variant
    assert direct_loaded.constbert_seq_length == model.constbert_seq_length
    assert torch.allclose(
        direct_loaded.document_projection_weight,
        loaded.document_projection_weight,
    )

    embeddings = loaded.encode(
        sentences=["hello world"],
        is_query=False,
        convert_to_tensor=True,
    )
    assert embeddings[0].shape[0] == loaded.constbert_seq_length
