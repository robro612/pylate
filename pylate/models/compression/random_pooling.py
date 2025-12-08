from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional

from .base import CompressionStrategy, CompressionStrategyConfigBase, CompressionArtifacts


@dataclass
class RandomPoolingConfig(CompressionStrategyConfigBase):
    """
    Randomly choose anchors for pooling non-protected tokens.
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
        return "random_pooling"


class RandomPoolingStrategy(CompressionStrategy):
    """
    Pool tokens by selecting random anchors and assigning tokens by cosine similarity.
    """

    required_artifacts: list[str] = []

    def __init__(self, config: RandomPoolingConfig):
        self.config = config

    @property
    def name(self) -> str:
        return (
            f"random-pool_r-{self.config.keep_ratio:.2f}"
            f"_p-{self.config.protected_tokens}"
            f"_min-{self.config.min_tokens}"
        )

    @property
    def strategy_type(self) -> str:
        return "random_pooling"

    def serialize(self) -> dict:
        return {
            "type": self.strategy_type,
            "config": self.config.serialize(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RandomPoolingStrategy":
        cfg = data.get("config", {}) or {}
        config = RandomPoolingConfig(
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
        rep_indices_per_doc: list[list[int]] = []

        for doc_idx, doc_emb in enumerate(embeddings):
            n_tokens = doc_emb.shape[0]
            protected = min(self.config.protected_tokens, n_tokens)
            n_nonprot = max(n_tokens - protected, 0)

            if n_nonprot == 0:
                pooled_embeddings.append(doc_emb)
                rep_indices_per_doc.append(list(range(n_tokens)))
                continue

            target_total = max(
                self.config.min_tokens,
                protected + int(self.config.keep_ratio * n_nonprot),
            )
            target_total = min(target_total, n_tokens)
            num_clusters = max(target_total - protected, 1)

            if num_clusters >= n_nonprot:
                pooled_embeddings.append(doc_emb)
                rep_indices_per_doc.append(list(range(n_tokens)))
                continue

            nonprot_emb = doc_emb[protected:, :]
            K = nonprot_emb.size(0)

            generator = torch.Generator(device=nonprot_emb.device)
            if self.config.seed is not None:
                generator.manual_seed(self.config.seed + doc_idx)

            perm = torch.randperm(K, generator=generator, device=nonprot_emb.device)
            anchor_local_idx = perm[:num_clusters]
            anchor_emb = nonprot_emb[anchor_local_idx]

            nonprot_norm = F.normalize(nonprot_emb, p=2, dim=-1)
            anchor_norm = F.normalize(anchor_emb, p=2, dim=-1)

            sims = nonprot_norm @ anchor_norm.t()
            cluster_labels = sims.argmax(dim=-1)  # [K]

            pooled_cluster_vectors: list[torch.Tensor] = []
            rep_indices: list[int] = list(range(protected))

            for cid in range(num_clusters):
                members = (cluster_labels == cid).nonzero(as_tuple=False).squeeze(-1)
                if members.numel() == 0:
                    continue
                pooled_vec = nonprot_emb[members].mean(dim=0)
                pooled_cluster_vectors.append(pooled_vec)

                # representative: first member
                first_member = members[0].item()
                rep_indices.append(protected + first_member)

            if pooled_cluster_vectors:
                pooled_doc = torch.cat(
                    [doc_emb[:protected], torch.stack(pooled_cluster_vectors, dim=0)],
                    dim=0,
                )
            else:
                pooled_doc = doc_emb[:protected]

            if pooled_doc.shape[0] < self.config.min_tokens and n_tokens >= self.config.min_tokens:
                pooled_doc = doc_emb
                rep_indices = list(range(n_tokens))

            pooled_embeddings.append(pooled_doc)
            rep_indices_per_doc.append(rep_indices)

        updated_artifacts: CompressionArtifacts = {}
        for name, value in artifacts.items():
            if not isinstance(value, list):
                updated_artifacts[name] = value
                continue

            pooled_artifacts_for_name: list[Any] = []

            for doc_idx, artifact_tokens in enumerate(value):
                rep_indices = rep_indices_per_doc[doc_idx]
                if isinstance(artifact_tokens, torch.Tensor):
                    length = artifact_tokens.size(0)
                    clipped = [i for i in rep_indices if i < length]
                    rep_idx_tensor = torch.tensor(
                        clipped,
                        device=artifact_tokens.device,
                        dtype=torch.long,
                    )
                    pooled_artifacts_for_name.append(artifact_tokens[rep_idx_tensor])
                else:
                    tokens_list = list(artifact_tokens)
                    pooled_artifacts_for_name.append(
                        [tokens_list[i] for i in rep_indices if i < len(tokens_list)]
                    )

            updated_artifacts[name] = pooled_artifacts_for_name

        return pooled_embeddings, updated_artifacts
