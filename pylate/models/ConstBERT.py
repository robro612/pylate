from __future__ import annotations

import json
import logging
import os
from typing import Literal, override

import torch
import torch.nn.functional as F
from torch import nn

from ..hf_hub.model_card import PylateModelCardData
from ..scores import SimilarityFunction
from .colbert import ColBERT

logger = logging.getLogger(__name__)


class ConstBERT(ColBERT):
    """
    ColBERT variant that projects token embeddings to a fixed sequence length.

    Projection variants:
    - "flatten": flatten (seq, hidden) -> linear -> reshape
    - "transpose": transpose to (hidden, seq) -> linear over seq -> transpose back
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        constbert_variant: Literal["flatten", "transpose"] | None = None,
        constbert_seq_length: int | None = None,
        # ColBERT parameters
        embedding_size: int | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        query_length: int | None = None,
        document_length: int | None = None,
        attend_to_expansion_tokens: bool | None = None,
        trust_remote_code: bool = False,
        do_query_expansion: bool | None = None,
        model_card_data: PylateModelCardData | None = None,
        similarity_fn_name: SimilarityFunction | str = SimilarityFunction.MAXSIM,
        **kwargs,
    ) -> None:
        config = None
        if model_name_or_path and os.path.isdir(model_name_or_path):
            config_path = os.path.join(
                model_name_or_path, "config_sentence_transformers.json"
            )
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                if constbert_variant is None:
                    constbert_variant = config.get("constbert_variant")
                if constbert_seq_length is None:
                    constbert_seq_length = config.get("constbert_seq_length")

        if constbert_variant is None:
            constbert_variant = "flatten"
        if constbert_seq_length is None:
            constbert_seq_length = 32

        super().__init__(
            model_name_or_path=model_name_or_path,
            embedding_size=embedding_size,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            query_length=query_length,
            document_length=document_length,
            attend_to_expansion_tokens=attend_to_expansion_tokens,
            trust_remote_code=trust_remote_code,
            skiplist_words=[],
            do_query_expansion=do_query_expansion,
            model_card_data=model_card_data,
            similarity_fn_name=similarity_fn_name,
            **kwargs,
        )

        if constbert_variant not in {"flatten", "transpose"}:
            raise ValueError(
                "constbert_variant must be 'flatten' or 'transpose', "
                f"got {constbert_variant!r}"
            )

        self.projection_variant = constbert_variant
        self.constbert_seq_length = constbert_seq_length
        self._projection_loaded = False

        embedding_dim = self._get_projection_embedding_dim()
        self._init_projection_parameters(embedding_dim)
        self._sync_projection_parameters()
        self._maybe_load_projection_weights(model_name_or_path, config)
        if self.device is not None:
            self.to(self.device)

        logger.info(
            "Initialized ConstBERT with variant=%s, doc_len=%s->%s",
            self.projection_variant,
            self.document_length,
            self.constbert_seq_length,
        )
        print(
            "ConstBERT init:",
            f"variant={self.projection_variant},",
            f"doc_len={self.document_length}->{self.constbert_seq_length},",
            f"embedding_dim={embedding_dim}",
        )

    def _get_projection_embedding_dim(self) -> int:
        if len(self) > 1 and hasattr(self[-1], "out_features"):
            return int(self[-1].out_features)
        return int(self[0].get_word_embedding_dimension())

    def _get_reference_param(self) -> torch.nn.Parameter | None:
        try:
            return next(self._first_module().auto_model.parameters())
        except StopIteration:
            return None

    def _init_projection_parameters(self, embedding_dim: int) -> None:
        self._validate_lengths(
            self.document_length, self.constbert_seq_length, "document"
        )

        if self.projection_variant == "flatten":
            doc_in = self.document_length * embedding_dim
            doc_out = self.constbert_seq_length * embedding_dim
        else:
            doc_in = self.document_length
            doc_out = self.constbert_seq_length

        ref_param = self._get_reference_param()
        device = ref_param.device if ref_param is not None else None
        dtype = ref_param.dtype if ref_param is not None else None

        self.document_projection_weight = nn.Parameter(
            torch.empty(doc_out, doc_in, device=device, dtype=dtype)
        )

        self.document_projection_bias = None

        nn.init.xavier_uniform_(self.document_projection_weight)

    def _maybe_load_projection_weights(
        self, model_name_or_path: str | None, config: dict | None
    ) -> None:
        if self._projection_loaded or not model_name_or_path:
            return
        if not os.path.isdir(model_name_or_path):
            return
        if config is not None:
            model_type = config.get("model_type")
            if model_type and model_type != "ConstBERT":
                return
            if model_type is None and not (
                "constbert_variant" in config or "constbert_seq_length" in config
            ):
                return
        self._load_projection_weights(model_name_or_path)

    def _load_projection_weights(self, model_path: str) -> None:
        if self._projection_loaded:
            return
        projection_path = os.path.join(model_path, "constbert_projection")
        state = None
        if os.path.exists(os.path.join(projection_path, "model.safetensors")):
            from safetensors.torch import load_file

            state = load_file(os.path.join(projection_path, "model.safetensors"))
        elif os.path.exists(os.path.join(projection_path, "pytorch_model.bin")):
            state = torch.load(
                os.path.join(projection_path, "pytorch_model.bin"),
                map_location="cpu",
            )

        if state is None:
            logger.warning(
                "No saved ConstBERT projection found at %s; using initialized weights.",
                projection_path,
            )
            return

        weight = state.get("document_weight")
        if weight is None:
            logger.warning(
                "ConstBERT projection state missing document_weight at %s.",
                projection_path,
            )
            return

        if weight.shape != self.document_projection_weight.shape:
            raise ValueError(
                "ConstBERT projection shape mismatch: "
                f"expected {tuple(self.document_projection_weight.shape)}, "
                f"got {tuple(weight.shape)}."
            )

        ref_param = self._get_reference_param()
        target_device = (
            ref_param.device if ref_param is not None else self.document_projection_weight.device
        )
        target_dtype = (
            ref_param.dtype if ref_param is not None else self.document_projection_weight.dtype
        )
        weight = weight.to(device=target_device, dtype=target_dtype)
        with torch.no_grad():
            self.document_projection_weight.copy_(weight)
        self._projection_loaded = True
        self._sync_projection_parameters()

    def _sync_projection_parameters(self) -> None:
        if not hasattr(self, "document_projection_weight"):
            return
        ref_param = self._get_reference_param()
        if ref_param is None:
            return
        if (
            self.document_projection_weight.device != ref_param.device
            or self.document_projection_weight.dtype != ref_param.dtype
        ):
            self.document_projection_weight.data = self.document_projection_weight.data.to(
                device=ref_param.device, dtype=ref_param.dtype
            )

    @staticmethod
    def _validate_lengths(
        input_length: int | None, output_length: int, label: str
    ) -> None:
        if input_length is None:
            raise ValueError(f"{label} length must be set for ConstBERT.")
        if output_length <= 0:
            raise ValueError(f"{label} output length must be > 0.")
        if output_length > input_length:
            raise ValueError(
                f"{label} output length ({output_length}) exceeds input length ({input_length})."
            )

    @override
    def tokenize(
        self,
        texts: list[str] | list[dict] | list[tuple[str, str]],
        is_query: bool = True,
        pad: bool = False,
    ) -> dict[str, torch.Tensor]:
        # Always pad documents to fixed length to match projection dimensions.
        if is_query:
            return super().tokenize(texts=texts, is_query=is_query, pad=pad)
        return super().tokenize(texts=texts, is_query=is_query, pad=True)

    def _project_embeddings(
        self,
        token_embeddings: torch.Tensor,
        projection_weight: torch.Tensor,
        projection_bias: torch.Tensor | None,
        output_length: int,
        expected_input_length: int,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        if seq_len != expected_input_length:
            raise ValueError(
                "Input sequence length mismatch for ConstBERT projection: "
                f"expected {expected_input_length}, got {seq_len}. "
                "Ensure tokenization pads to the configured length."
            )

        if self.projection_variant == "flatten":
            flattened = token_embeddings.reshape(batch_size, seq_len * hidden_dim)
            projected = F.linear(flattened, projection_weight, projection_bias)
            return projected.view(batch_size, output_length, hidden_dim)

        transposed = token_embeddings.transpose(1, 2)  # (batch, hidden, seq)
        projected = F.linear(transposed, projection_weight, projection_bias)
        return projected.transpose(1, 2)

    @override
    def forward(
        self,
        features: dict[str, torch.Tensor] = None,
        is_query: bool | None = None,
        input: dict[str, torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if features is None and input is not None:
            features = input
        elif features is None and input is None:
            raise ValueError("Either 'features' or 'input' must be provided")

        if is_query is None:
            is_query = True

        if is_query:
            return super().forward(features, is_query=True)

        out_features = super().forward(features, is_query=False)
        token_embeddings = out_features["token_embeddings"]
        self._sync_projection_parameters()

        projected = self._project_embeddings(
            token_embeddings=token_embeddings,
            projection_weight=self.document_projection_weight,
            projection_bias=self.document_projection_bias,
            output_length=self.constbert_seq_length,
            expected_input_length=self.document_length,
        )
        output_length = self.constbert_seq_length

        out_features["token_embeddings"] = projected
        out_features["attention_mask"] = projected.new_ones(
            (projected.size(0), output_length), dtype=features["attention_mask"].dtype
        )

        if "input_ids" in features:
            features["input_ids"] = features["input_ids"][:, :output_length]
            out_features["input_ids"] = features["input_ids"]

        if "token_type_ids" in features:
            features["token_type_ids"] = features["token_type_ids"][:, :output_length]
            out_features["token_type_ids"] = features["token_type_ids"]

        if "sentence_embedding" in out_features:
            out_features["sentence_embedding"] = projected.mean(dim=1)

        return out_features

    def to(self, *args, **kwargs) -> "ConstBERT":
        model = super().to(*args, **kwargs)
        if hasattr(self, "document_projection_weight"):
            self._sync_projection_parameters()
        return model

    @override
    def save(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list | None = None,
        safe_serialization: bool = True,
    ) -> None:
        super().save(
            path=path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )

        projection_path = os.path.join(path, "constbert_projection")
        os.makedirs(projection_path, exist_ok=True)

        projection_state = {
            "document_weight": self.document_projection_weight.detach().cpu(),
        }
        if safe_serialization:
            from safetensors.torch import save_file

            save_file(
                projection_state,
                os.path.join(projection_path, "model.safetensors"),
            )
        else:
            torch.save(
                projection_state, os.path.join(projection_path, "pytorch_model.bin")
            )

        config_path = os.path.join(path, "config_sentence_transformers.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        config["model_type"] = "ConstBERT"
        config["constbert_variant"] = self.projection_variant
        config["constbert_seq_length"] = self.constbert_seq_length

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved ConstBERT to %s", path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "ConstBERT":
        config_path = os.path.join(path, "config_sentence_transformers.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        const_params = {
            "constbert_variant": config.get("constbert_variant", "flatten"),
            "constbert_seq_length": config.get("constbert_seq_length", 32),
        }

        merged_params = {**const_params, **kwargs}
        model = cls(model_name_or_path=path, **merged_params)
        model._load_projection_weights(path)
        return model
