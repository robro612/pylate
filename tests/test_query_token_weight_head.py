from __future__ import annotations

import torch
from torch import nn

from pylate.models import Dense, QueryTokenWeightHead


def test_query_token_weight_head_normalizes_with_mask() -> None:
    head = QueryTokenWeightHead(
        layers=[nn.Linear(4, 1, bias=False)],
        normalization_mode="sum1",
        positive_activation="softplus",
    )
    with torch.no_grad():
        head.layers[0].weight.fill_(1.0)

    token_embeddings = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]]
    )
    attention_mask = torch.tensor([[1, 1, 0]])
    features = {"token_embeddings": token_embeddings, "attention_mask": attention_mask}
    outputs = head(features)

    weights = outputs["query_token_weights"]
    assert weights.shape == (1, 3)
    torch.testing.assert_close(weights[:, 2], torch.zeros(1))
    torch.testing.assert_close(weights[:, :2].sum(dim=1), torch.ones(1), atol=1e-6, rtol=0.0)


def test_query_token_weight_head_supports_dense_layers() -> None:
    head = QueryTokenWeightHead(
        layers=[
            Dense(in_features=4, out_features=4, activation_function=nn.ReLU()),
            Dense(in_features=4, out_features=1),
        ],
        normalization_mode="none",
        positive_activation="relu",
    )
    token_embeddings = torch.randn(2, 5, 4)
    outputs = head({"token_embeddings": token_embeddings})
    weights = outputs["query_token_weights"]

    assert weights.shape == (2, 5)
    assert torch.all(weights >= 0)
