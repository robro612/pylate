from __future__ import annotations

import torch

from pylate import models


def test_memory_token_forward_shapes_and_masks() -> None:
    model = models.MemoryTokenColBERT(
        model_name_or_path="bert-base-uncased",
        num_memory_tokens=4,
        attend_to_memory_tokens=False,
        document_length=16,
        query_length=8,
    )

    documents = ["hello world", "quick brown fox"]
    features = model.tokenize(texts=documents, is_query=False, pad=True)

    assert features["input_ids"].shape[1] == model.document_length + model.num_memory_tokens
    assert torch.all(features["attention_mask"][:, -model.num_memory_tokens:] == 0)

    features = {k: v.to(model.device) for k, v in features.items()}
    with torch.no_grad():
        out_features = model.forward(input=features, is_query=False)

    assert out_features["token_embeddings"].shape[1] == model.num_memory_tokens
    assert out_features["attention_mask"].shape[1] == model.num_memory_tokens
    assert torch.all(out_features["attention_mask"] == 1)


def test_memory_token_encode_shapes() -> None:
    model = models.MemoryTokenColBERT(
        model_name_or_path="bert-base-uncased",
        num_memory_tokens=4,
        attend_to_memory_tokens=True,
        document_length=16,
        query_length=8,
    )

    documents = ["hello world", "quick brown fox"]
    embeddings = model.encode(
        sentences=documents,
        is_query=False,
        convert_to_numpy=False,
        convert_to_tensor=False,
    )

    assert len(embeddings) == len(documents)
    for embedding in embeddings:
        assert embedding.shape[0] == model.num_memory_tokens


def test_memory_token_save_and_load(tmp_path) -> None:
    model = models.MemoryTokenColBERT(
        model_name_or_path="bert-base-uncased",
        num_memory_tokens=4,
        attend_to_memory_tokens=True,
        document_length=16,
        query_length=8,
    )

    output_dir = tmp_path / "memory_model"
    model.save_pretrained(str(output_dir))

    loaded = models.MemoryTokenColBERT.load(str(output_dir))
    assert loaded.num_memory_tokens == model.num_memory_tokens
    assert loaded.attend_to_memory_tokens == model.attend_to_memory_tokens
    assert loaded.memory_token_prefix == model.memory_token_prefix

    embeddings = loaded.encode(
        sentences=["hello world"],
        is_query=False,
        convert_to_numpy=False,
        convert_to_tensor=False,
    )
    assert embeddings[0].shape[0] == loaded.num_memory_tokens

