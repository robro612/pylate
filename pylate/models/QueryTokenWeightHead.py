from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Literal

import torch
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from sentence_transformers.util import import_from_string
from torch import nn

from .Dense import Dense

__all__ = ["QueryTokenWeightHead"]


class QueryTokenWeightHead(nn.Module):
    """Predict and normalize per-token query weights.

    This module is inserted in the SentenceTransformer module chain and stores
    query weights in ``features["query_token_weights"]`` while preserving
    ``features["token_embeddings"]`` unchanged.
    """

    def __init__(
        self,
        layers: Iterable[nn.Module],
        normalization_mode: Literal["none", "sum1", "mean1"] = "sum1",
        positive_activation: str = "softplus",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(list(layers))
        if not self.layers:
            raise ValueError("QueryTokenWeightHead requires at least one layer.")
        self.normalization_mode = normalization_mode
        self.positive_activation = positive_activation
        self.eps = eps

    def _apply_positive_activation(self, weights: torch.Tensor) -> torch.Tensor:
        if self.positive_activation == "softplus":
            return torch.nn.functional.softplus(weights)
        if self.positive_activation == "relu":
            return torch.relu(weights)
        if self.positive_activation == "exp":
            return torch.exp(weights)
        if self.positive_activation == "identity":
            return weights
        raise ValueError(
            f"Unsupported positive_activation: {self.positive_activation}. "
            "Use one of: softplus, relu, exp, identity."
        )

    @staticmethod
    def _apply_layer(layer: nn.Module, token_embeddings: torch.Tensor) -> torch.Tensor:
        if isinstance(layer, Dense):
            return layer({"token_embeddings": token_embeddings})["token_embeddings"]

        output = layer(token_embeddings)
        if isinstance(output, dict):
            return output["token_embeddings"]
        return output

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_embeddings = features["token_embeddings"]
        scores = token_embeddings
        for layer in self.layers:
            scores = self._apply_layer(layer=layer, token_embeddings=scores)

        if scores.ndim == token_embeddings.ndim and scores.size(-1) == 1:
            scores = scores.squeeze(-1)
        if scores.ndim != token_embeddings.ndim - 1:
            raise ValueError(
                "QueryTokenWeightHead output must have shape (batch, tokens) or "
                "(batch, tokens, 1)."
            )

        weights = self._apply_positive_activation(scores)

        attention_mask = features.get("attention_mask", None)
        if attention_mask is not None:
            mask = attention_mask.to(dtype=weights.dtype)
            weights = weights * mask
        else:
            mask = None

        if self.normalization_mode == "sum1":
            normalizer = weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
            weights = weights / normalizer
            if mask is not None:
                weights = weights * mask
        elif self.normalization_mode == "mean1":
            if mask is not None:
                counts = mask.sum(dim=1, keepdim=True).clamp_min(self.eps)
            else:
                counts = torch.full_like(
                    weights.sum(dim=1, keepdim=True),
                    fill_value=weights.size(1),
                )
            means = (weights.sum(dim=1, keepdim=True) / counts).clamp_min(self.eps)
            weights = weights / means
            if mask is not None:
                weights = weights * mask
        elif self.normalization_mode != "none":
            raise ValueError(
                f"Unsupported normalization_mode: {self.normalization_mode}. "
                "Use one of: none, sum1, mean1."
            )

        features["query_token_weights"] = weights
        return features

    def get_config_dict(self) -> dict:
        return {
            "normalization_mode": self.normalization_mode,
            "positive_activation": self.positive_activation,
            "eps": self.eps,
        }

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(self.get_config_dict(), f)

        modules = {
            str(i): layer.get_config_dict() if hasattr(layer, "get_config_dict") else {}
            for i, layer in enumerate(self.layers)
        }
        with open(os.path.join(output_path, "layers.json"), "w") as f:
            json.dump(modules, f)

        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path: str) -> "QueryTokenWeightHead":
        with open(os.path.join(input_path, "config.json")) as f:
            config = json.load(f)

        with open(os.path.join(input_path, "layers.json")) as f:
            layer_configs = json.load(f)

        layers = []
        for _, layer_config in sorted(layer_configs.items(), key=lambda item: int(item[0])):
            if "in_features" in layer_config and "out_features" in layer_config:
                if "activation_function" in layer_config and isinstance(
                    layer_config["activation_function"], str
                ):
                    layer_config["activation_function"] = import_from_string(
                        layer_config["activation_function"]
                    )()
                layers.append(Dense(**layer_config))
            else:
                raise ValueError(
                    "Unsupported layer in QueryTokenWeightHead.load. "
                    "Only Dense-based heads are currently loadable."
                )

        model = QueryTokenWeightHead(layers=layers, **config)
        safetensor_path = os.path.join(input_path, "model.safetensors")
        if os.path.exists(safetensor_path):
            load_safetensors_model(model, safetensor_path)
            return model

        model.load_state_dict(
            torch.load(
                os.path.join(input_path, "pytorch_model.bin"),
                map_location=torch.device("cpu"),
            )
        )
        return model
