from .base import *

@dataclass
class AttentionPoolingConfig(CompressionStrategyConfigBase):
    """
    Configuration for attention-based pooling.

    Selects anchor tokens using the highest attention scores, then pools
    remaining tokens to their most similar anchor based on cosine similarity.

    Attributes
    ----------
    keep_ratio
        Ratio of tokens to keep as anchors (after protected tokens).
        E.g., 0.5 means keep 50% of non-protected tokens as anchors.
    protected_tokens
        Number of leading tokens to always retain (CLS / prefixes).
    min_tokens
        Minimum number of total tokens to keep after pooling.
    show_progress_bar
        If True, shows a progress bar during pooling. Defaults to False.
    """

    keep_ratio: float = 0.5
    protected_tokens: int = 1
    min_tokens: int = 8
    show_progress_bar: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1].")
        if self.protected_tokens < 0:
            raise ValueError("protected_tokens must be >= 0.")
        if self.min_tokens <= 0:
            raise ValueError("min_tokens must be > 0.")

    def serialize(self) -> dict:
        """
        Serialize this configuration to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "keep_ratio": self.keep_ratio,
            "protected_tokens": self.protected_tokens,
            "min_tokens": self.min_tokens,
            "show_progress_bar": self.show_progress_bar,
        }

    @property
    def strategy_type(self) -> str:
        return "attention_pooling"


class AttentionPoolingStrategy(CompressionStrategy):
    """
    Attention-based pooling strategy.
    
    Selects anchor tokens using the highest attention scores from the model's
    last layer, then pools remaining tokens to their most similar anchor based
    on cosine similarity.
    
    Algorithm:
    1. Protect the first `protected_tokens` tokens
    2. Select anchors from remaining tokens with highest attention scores
    3. For each non-anchor token, find the most similar anchor (cosine similarity)
    4. Pool tokens to their assigned anchor by averaging embeddings
    """

    required_artifacts: list[str] = ["attention_scores"]

    def __init__(self, config: AttentionPoolingConfig):
        self.config = config

    @property
    def name(self) -> str:
        return (
            f"attention-pool_r-{self.config.keep_ratio:.2f}"
            f"_p-{self.config.protected_tokens}"
            f"_min-{self.config.min_tokens}"
        )

    @property
    def strategy_type(self) -> str:
        return "attention_pooling"

    def serialize(self) -> dict:
        return {
            "type": self.strategy_type,
            "config": self.config.serialize(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttentionPoolingStrategy":
        config_data = data.get("config", {})
        config = AttentionPoolingConfig(
            keep_ratio=config_data.get("keep_ratio", 0.5),
            protected_tokens=config_data.get("protected_tokens", 1),
            min_tokens=config_data.get("min_tokens", 8),
            show_progress_bar=config_data.get("show_progress_bar", False),
        )
        return cls(config)

    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply attention-based pooling to embeddings.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain "attention_scores" as a list of tensors.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pooled_embeddings, updated_artifacts)
        
        Raises
        ------
        ValueError
            If attention_scores artifact is missing
        """
        if "attention_scores" not in artifacts:
            raise ValueError(
                "AttentionPoolingStrategy requires 'attention_scores' artifact. "
                "Ensure attention_scores are provided when encoding."
            )

        attention_scores = artifacts["attention_scores"]
        if not isinstance(attention_scores, list) or len(attention_scores) != len(embeddings):
            raise ValueError("attention_scores must be a list matching embeddings length")

        pooled_embeddings = []
        doc_mappings = []

        iterator = tqdm(
            enumerate(zip(embeddings, attention_scores)),
            desc=f"Attention pooling (keep_ratio={self.config.keep_ratio})",
            total=len(embeddings),
            disable=not self.config.show_progress_bar,
        )

        for doc_idx, (doc_emb, doc_attention) in iterator:
            device = doc_emb.device
            n_tokens, dim = doc_emb.shape

            if n_tokens == 0:
                pooled_embeddings.append(doc_emb)
                doc_mappings.append(
                    {"protected": 0, "cluster_labels": [], "num_clusters": 0}
                )
                continue

            protected = min(self.config.protected_tokens, n_tokens)
            n_non_protected = max(n_tokens - protected, 0)

            if n_non_protected <= 0:
                # Nothing to pool
                pooled_embeddings.append(doc_emb)
                doc_mappings.append(
                    {"protected": protected, "cluster_labels": [], "num_clusters": 0}
                )
                continue

            # Target number of total tokens after pooling
            target_total = max(
                self.config.min_tokens,
                protected + int(self.config.keep_ratio * n_non_protected),
            )
            num_clusters = max(1, target_total - protected)

            # Split into protected and non-protected
            prot_emb = doc_emb[:protected, :]
            nonprot_emb = doc_emb[protected:, :]
            nonprot_attention = doc_attention[protected:]

            # Select anchors: tokens with highest attention scores
            _, anchor_local_idx = torch.topk(
                nonprot_attention, k=min(num_clusters, n_non_protected), largest=True, sorted=False
            )
            # Sort anchors by position to maintain document order
            anchor_local_idx, _ = torch.sort(anchor_local_idx)

            anchor_emb = nonprot_emb[anchor_local_idx]  # [C, dim]

            # Assign each token to nearest anchor (cosine similarity)
            import torch.nn.functional as F
            nonprot_norm = F.normalize(nonprot_emb, p=2, dim=-1)
            anchor_norm = F.normalize(anchor_emb, p=2, dim=-1)
            sims = nonprot_norm @ anchor_norm.t()  # [K, C]
            cluster_labels_tensor = sims.argmax(dim=-1)  # [K]
            cluster_labels = cluster_labels_tensor.cpu().tolist()

            # Pool embeddings for each cluster
            pooled_cluster_vectors: list[torch.Tensor] = []
            for cid in range(num_clusters):
                mask = cluster_labels_tensor == cid
                if not mask.any():
                    continue
                members = nonprot_emb[mask]  # [num_members, dim]
                pooled_vec = members.mean(dim=0)
                pooled_cluster_vectors.append(pooled_vec)

            num_clusters_effective = len(pooled_cluster_vectors)
            if num_clusters_effective == 0:
                # Fallback: no pooling
                pooled_embeddings.append(doc_emb)
                doc_mappings.append(
                    {"protected": protected, "cluster_labels": cluster_labels, "num_clusters": 0}
                )
                continue

            # Combine protected + pooled
            pooled_cluster_stacked = torch.stack(pooled_cluster_vectors, dim=0)
            final_emb = torch.cat([prot_emb, pooled_cluster_stacked], dim=0)
            pooled_embeddings.append(final_emb)

            doc_mappings.append(
                {
                    "protected": protected,
                    "cluster_labels": cluster_labels,
                    "num_clusters": num_clusters_effective,
                }
            )

        # Update artifacts
        updated_artifacts = self._update_artifacts(
            artifacts, embeddings, pooled_embeddings, doc_mappings
        )

        return pooled_embeddings, updated_artifacts

    def _update_artifacts(
        self,
        artifacts: CompressionArtifacts,
        original_embeddings: list[torch.Tensor],
        pooled_embeddings: list[torch.Tensor],
        doc_mappings: list[dict],
    ) -> CompressionArtifacts:
        """Update artifacts to match pooled embeddings."""
        updated_artifacts = {}

        for artifact_name, artifact_value in artifacts.items():
            if artifact_name == "attention_scores":
                # Pool attention scores by averaging within clusters
                pooled_attention = []
                for doc_idx, doc_attention in enumerate(artifact_value):
                    mapping = doc_mappings[doc_idx]
                    protected = mapping["protected"]
                    cluster_labels = mapping["cluster_labels"]
                    num_clusters = mapping["num_clusters"]

                    if num_clusters == 0:
                        pooled_attention.append(doc_attention)
                        continue

                    prot_attention = doc_attention[:protected]
                    nonprot_attention = doc_attention[protected:]

                    # Average attention scores within each cluster
                    cluster_attention_list = []
                    cluster_labels_tensor = torch.tensor(
                        cluster_labels, device=doc_attention.device
                    )
                    for cid in range(num_clusters):
                        mask = cluster_labels_tensor == cid
                        if not mask.any():
                            continue
                        cluster_avg = nonprot_attention[mask].mean()
                        cluster_attention_list.append(cluster_avg)

                    if cluster_attention_list:
                        cluster_attention = torch.stack(cluster_attention_list)
                        final_attention = torch.cat([prot_attention, cluster_attention])
                    else:
                        final_attention = prot_attention

                    pooled_attention.append(final_attention)

                updated_artifacts[artifact_name] = pooled_attention

            elif isinstance(artifact_value, list) and len(artifact_value) == len(original_embeddings):
                # Shape-matched artifact: apply same pooling
                pooled_artifacts = []
                for doc_idx, artifact_tokens in enumerate(artifact_value):
                    mapping = doc_mappings[doc_idx]
                    protected = mapping["protected"]
                    cluster_labels = mapping["cluster_labels"]
                    num_clusters = mapping["num_clusters"]

                    if num_clusters == 0:
                        pooled_artifacts.append(artifact_tokens)
                        continue

                    # Check if artifact is a tensor or list
                    is_tensor = isinstance(artifact_tokens, torch.Tensor)
                    if is_tensor:
                        n_tokens_art = artifact_tokens.size(0)
                    else:
                        n_tokens_art = len(artifact_tokens)

                    # Protected tokens
                    if is_tensor:
                        prot_tokens = artifact_tokens[:protected]
                    else:
                        prot_tokens = artifact_tokens[:protected]

                    # For each cluster, pick the first member's token as representative
                    pooled_cluster_tokens = []
                    for cid in range(num_clusters):
                        member_indices = [
                            i for i, lab in enumerate(cluster_labels) if lab == cid
                        ]
                        if not member_indices:
                            continue
                        first_local = member_indices[0]
                        orig_idx = protected + first_local

                        if orig_idx >= n_tokens_art:
                            continue

                        if is_tensor:
                            pooled_cluster_tokens.append(artifact_tokens[orig_idx])
                        else:
                            pooled_cluster_tokens.append(artifact_tokens[orig_idx])

                    # Combine protected + pooled
                    if is_tensor and pooled_cluster_tokens:
                        pooled_cluster_stacked = torch.stack(pooled_cluster_tokens)
                        final_tokens = torch.cat([prot_tokens, pooled_cluster_stacked])
                    elif is_tensor:
                        final_tokens = prot_tokens
                    else:
                        final_tokens = list(prot_tokens) + pooled_cluster_tokens

                    pooled_artifacts.append(final_tokens)

                updated_artifacts[artifact_name] = pooled_artifacts
            else:
                # Metadata artifact: pass through unchanged
                updated_artifacts[artifact_name] = artifact_value

        return updated_artifacts

