from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any, Optional

from .base import CompressionStrategy, CompressionStrategyConfigBase, CompressionArtifacts


@dataclass
class RandomPruningConfig(CompressionStrategyConfigBase):
    """
    Randomly prune non-protected tokens while preserving a minimum length.
    """

    keep_ratio: float = 0.5
    protected_tokens: int = 1
    min_tokens: int = 8
    seed: Optional[int] = 42

    def __post_init__(self) -> None:
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1].")
        if self.protected_tokens < 0:
            raise ValueError("protected_tokens must be >= 0.")
        if self.min_tokens <= 0:
            raise ValueError("min_tokens must be > 0.")

    def serialize(self) -> dict:
        return {
            "keep_ratio": self.keep_ratio,
            "protected_tokens": self.protected_tokens,
            "min_tokens": self.min_tokens,
            "seed": self.seed,
        }

    @property
    def strategy_type(self) -> str:
        return "random_pruning"


class RandomPruningStrategy(CompressionStrategy):
    """
    Randomly drop non-protected tokens. Useful as a baseline for stochastic pruning.
    """

    required_artifacts: list[str] = []

    def __init__(self, config: RandomPruningConfig):
        self.config = config

    @property
    def name(self) -> str:
        return (
            f"random-prune_r-{self.config.keep_ratio:.2f}"
            f"_p-{self.config.protected_tokens}"
            f"_min-{self.config.min_tokens}"
        )

    @property
    def strategy_type(self) -> str:
        return "random_pruning"

    def serialize(self) -> dict:
        return {
            "type": self.strategy_type,
            "config": self.config.serialize(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RandomPruningStrategy":
        cfg = data.get("config", {}) or {}
        config = RandomPruningConfig(
            keep_ratio=cfg.get("keep_ratio", 0.5),
            protected_tokens=cfg.get("protected_tokens", 1),
            min_tokens=cfg.get("min_tokens", 8),
            seed=cfg.get("seed", 42),
        )
        return cls(config)

    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        pooled_embeddings: list[torch.Tensor] = []
        kept_indices_per_doc: list[list[int]] = []

        for doc_idx, doc_emb in enumerate(embeddings):
            n_tokens = doc_emb.shape[0]
            protected = min(self.config.protected_tokens, n_tokens)
            n_nonprot = max(n_tokens - protected, 0)

            if n_nonprot == 0:
                pooled_embeddings.append(doc_emb)
                kept_indices_per_doc.append(list(range(n_tokens)))
                continue

            target_total = max(
                self.config.min_tokens,
                protected + int(self.config.keep_ratio * n_nonprot),
            )
            target_total = min(target_total, n_tokens)

            if target_total >= n_tokens:
                pooled_embeddings.append(doc_emb)
                kept_indices_per_doc.append(list(range(n_tokens)))
                continue

            keep_nonprot = max(target_total - protected, 0)

            device = doc_emb.device
            generator = torch.Generator(device=device)
            if self.config.seed is not None:
                generator.manual_seed(self.config.seed + doc_idx)

            perm = torch.randperm(n_nonprot, generator=generator, device=device)
            chosen = perm[:keep_nonprot].cpu().tolist()
            chosen_sorted = sorted(chosen)
            keep_indices = list(range(protected)) + [protected + i for i in chosen_sorted]

            new_emb = doc_emb[keep_indices]
            pooled_embeddings.append(new_emb)
            kept_indices_per_doc.append(keep_indices)

        updated_artifacts: CompressionArtifacts = {}
        for name, value in artifacts.items():
            if not isinstance(value, list):
                updated_artifacts[name] = value
                continue

            pooled_artifacts_for_name: list[Any] = []
            for doc_idx, artifact_tokens in enumerate(value):
                keep_indices = kept_indices_per_doc[doc_idx]
                if isinstance(artifact_tokens, torch.Tensor):
                    length = artifact_tokens.size(0)
                    clipped = [i for i in keep_indices if i < length]
                    rep_idx_tensor = torch.tensor(
                        clipped,
                        device=artifact_tokens.device,
                        dtype=torch.long,
                    )
                    pooled_artifacts_for_name.append(artifact_tokens[rep_idx_tensor])
                else:
                    tokens_list = list(artifact_tokens)
                    pooled_artifacts_for_name.append(
                        [tokens_list[i] for i in keep_indices if i < len(tokens_list)]
                    )

            updated_artifacts[name] = pooled_artifacts_for_name

        return pooled_embeddings, updated_artifacts
