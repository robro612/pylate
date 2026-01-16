from __future__ import annotations

import json
import logging
import os
from typing import Iterable, override

import torch
from torch import nn

from ..hf_hub.model_card import PylateModelCardData
from ..scores import SimilarityFunction
from .colbert import ColBERT

logger = logging.getLogger(__name__)


class MemoryTokenColBERT(ColBERT):
    """
    ColBERT with fixed memory tokens appended to documents.

    - At encode time, K memory tokens are appended to document inputs.
    - At inference time, only memory token embeddings are returned.
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        num_memory_tokens: int = 32,
        memory_token_prefix: str = "mem_",
        attend_to_memory_tokens: bool = False,
        # ColBERT parameters
        embedding_size: int | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        query_length: int | None = None,
        document_length: int | None = None,
        attend_to_expansion_tokens: bool | None = None,
        trust_remote_code: bool = False,
        skiplist_words: Iterable[str] | None = None,
        do_query_expansion: bool | None = None,
        model_card_data: PylateModelCardData | None = None,
        similarity_fn_name: SimilarityFunction | str = SimilarityFunction.MAXSIM,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            embedding_size=embedding_size,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            query_length=query_length,
            document_length=document_length,
            attend_to_expansion_tokens=attend_to_expansion_tokens,
            trust_remote_code=trust_remote_code,
            skiplist_words=skiplist_words,
            do_query_expansion=do_query_expansion,
            model_card_data=model_card_data,
            similarity_fn_name=similarity_fn_name,
            **kwargs,
        )

        self.num_memory_tokens = num_memory_tokens
        self.memory_token_prefix = memory_token_prefix
        self.attend_to_memory_tokens = attend_to_memory_tokens

        self._memory_tokens = [
            f"[${self.memory_token_prefix}{i}]"
            for i in range(1, self.num_memory_tokens + 1)
        ]
        self.memory_token_ids: list[int] = []
        self._add_memory_tokens()

    def _add_memory_tokens(self) -> None:
        if self.num_memory_tokens <= 0:
            self.memory_token_ids = []
            return

        try:
            self._first_module().auto_model.resize_token_embeddings(len(self.tokenizer))
            added = self.tokenizer.add_tokens(self._memory_tokens)
            if added > 0:
                self._first_module().auto_model.resize_token_embeddings(len(self.tokenizer))
        except NotImplementedError:
            logger.warning(
                "Tokenizer does not support resizing embeddings; memory tokens may map to UNK."
            )

        self.memory_token_ids = [
            self.tokenizer.convert_tokens_to_ids(token) for token in self._memory_tokens
        ]

        if self.tokenizer.unk_token_id in self.memory_token_ids:
            logger.warning(
                "Some memory tokens mapped to UNK. Check tokenizer resizing support."
            )

    @override
    def tokenize(
        self,
        texts: list[str] | list[dict] | list[tuple[str, str]],
        is_query: bool = True,
        pad: bool = False,
    ) -> dict[str, torch.Tensor]:
        tokenized_outputs = super().tokenize(texts=texts, is_query=is_query, pad=pad)

        if is_query or self.num_memory_tokens <= 0:
            return tokenized_outputs

        batch_size = tokenized_outputs["input_ids"].size(0)
        memory_ids = tokenized_outputs["input_ids"].new_tensor(self.memory_token_ids)
        memory_ids = memory_ids.unsqueeze(0).expand(batch_size, -1)

        tokenized_outputs["input_ids"] = torch.cat(
            [tokenized_outputs["input_ids"], memory_ids], dim=1
        )

        attend_value = 1 if self.attend_to_memory_tokens else 0
        memory_attention_mask = tokenized_outputs["attention_mask"].new_full(
            (batch_size, self.num_memory_tokens),
            attend_value,
        )
        tokenized_outputs["attention_mask"] = torch.cat(
            [tokenized_outputs["attention_mask"], memory_attention_mask], dim=1
        )

        if "token_type_ids" in tokenized_outputs:
            memory_token_type_ids = tokenized_outputs["token_type_ids"].new_full(
                (batch_size, self.num_memory_tokens), 0
            )
            tokenized_outputs["token_type_ids"] = torch.cat(
                [tokenized_outputs["token_type_ids"], memory_token_type_ids], dim=1
            )

        return tokenized_outputs

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

        if is_query or self.num_memory_tokens <= 0:
            return super().forward(features, is_query=True)

        out_features = super().forward(features, is_query=False)

        token_embeddings = out_features["token_embeddings"]
        if token_embeddings.size(1) < self.num_memory_tokens:
            raise ValueError(
                "Input sequence is shorter than num_memory_tokens; "
                "did you forget to append memory tokens?"
            )

        memory_embeddings = token_embeddings[:, -self.num_memory_tokens :, :]
        out_features["token_embeddings"] = memory_embeddings

        batch_size = memory_embeddings.size(0)
        memory_attention_mask = memory_embeddings.new_ones(
            (batch_size, self.num_memory_tokens),
            dtype=features["attention_mask"].dtype,
        )
        out_features["attention_mask"] = memory_attention_mask

        if "input_ids" in features:
            features["input_ids"] = features["input_ids"][:, -self.num_memory_tokens :]
            out_features["input_ids"] = features["input_ids"]

        if "token_type_ids" in features:
            features["token_type_ids"] = features["token_type_ids"][
                :, -self.num_memory_tokens :
            ]

        if "sentence_embedding" in out_features:
            out_features["sentence_embedding"] = memory_embeddings.mean(dim=1)

        return out_features

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

        config_path = os.path.join(path, "config_sentence_transformers.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        config["model_type"] = "MemoryTokenColBERT"
        config["num_memory_tokens"] = self.num_memory_tokens
        config["memory_token_prefix"] = self.memory_token_prefix
        config["attend_to_memory_tokens"] = self.attend_to_memory_tokens

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved MemoryTokenColBERT to %s", path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "MemoryTokenColBERT":
        config_path = os.path.join(path, "config_sentence_transformers.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        memory_params = {
            "num_memory_tokens": config.get("num_memory_tokens", 32),
            "memory_token_prefix": config.get("memory_token_prefix", "mem_"),
            "attend_to_memory_tokens": config.get("attend_to_memory_tokens", False),
        }

        merged_params = {**memory_params, **kwargs}
        return cls(model_name_or_path=path, **merged_params)

