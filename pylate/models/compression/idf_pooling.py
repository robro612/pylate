"""
IDF-based pooling strategy for ColBERT compression.

This strategy selects anchor tokens based on IDF scores (high IDF = informative = anchors),
then pools remaining tokens to their most similar anchor based on cosine similarity.
"""

from .base import *
from ..utils import TokenTFIDFStats


@dataclass
class IDFPoolingConfig(CompressionStrategyConfigBase):
    """
    Configuration for IDF-based pooling.

    Selects anchor tokens using the highest IDF scores, then pools
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
    use_tfidf
        If True, uses TF-IDF scoring (considers in-document frequency).
        If False, uses only IDF scoring. Defaults to False.
    ignore_token_ids
        Set of token IDs to ignore when computing IDF scores (e.g., special tokens).
    show_progress_bar
        If True, shows a progress bar during pooling. Defaults to False.
    """

    keep_ratio: float = 0.5
    protected_tokens: int = 1
    min_tokens: int = 8
    use_tfidf: bool = False
    ignore_token_ids: Optional[set[int]] = None
    show_progress_bar: bool = False

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
            "use_tfidf": self.use_tfidf,
            "ignore_token_ids": list(self.ignore_token_ids) if self.ignore_token_ids else None,
            "show_progress_bar": self.show_progress_bar,
        }

    @property
    def strategy_type(self) -> str:
        return "idf_pooling"


class IDFPoolingStrategy(CompressionStrategy):
    """
    IDF-based pooling strategy.
    
    Selects anchor tokens using the highest IDF scores (most informative tokens),
    then pools remaining tokens to their most similar anchor based on cosine similarity.
    
    Algorithm:
    1. Compute IDF/TF-IDF statistics from input_ids
    2. Protect the first `protected_tokens` tokens
    3. Select anchors from remaining tokens with highest IDF/TF-IDF scores
    4. For each non-anchor token, find the most similar anchor (cosine similarity)
    5. Pool tokens to their assigned anchor by averaging embeddings
    """

    required_artifacts: list[str] = ["input_ids"]

    def __init__(self, config: IDFPoolingConfig):
        self.config = config

    @property
    def name(self) -> str:
        tfidf_str = "_tfidf" if self.config.use_tfidf else ""
        return (
            f"idf-pool{tfidf_str}_r-{self.config.keep_ratio:.2f}"
            f"_p-{self.config.protected_tokens}"
            f"_min-{self.config.min_tokens}"
        )

    @property
    def strategy_type(self) -> str:
        return "idf_pooling"

    def serialize(self) -> dict:
        return {
            "type": self.strategy_type,
            "config": self.config.serialize(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IDFPoolingStrategy":
        config_data = data.get("config", {})
        config = IDFPoolingConfig(
            keep_ratio=config_data.get("keep_ratio", 0.5),
            protected_tokens=config_data.get("protected_tokens", 1),
            min_tokens=config_data.get("min_tokens", 8),
            use_tfidf=config_data.get("use_tfidf", False),
            ignore_token_ids=set(config_data.get("ignore_token_ids", [])) if config_data.get("ignore_token_ids") else None,
            show_progress_bar=config_data.get("show_progress_bar", False),
        )
        return cls(config)

    def _compute_tfidf_stats(self, input_ids: list[torch.Tensor]) -> TokenTFIDFStats:
        """Compute TF-IDF statistics from input_ids."""
        tokenized_docs = [doc_input_ids.cpu().tolist() for doc_input_ids in input_ids]
        stats = TokenTFIDFStats(num_docs=len(tokenized_docs))
        stats.fit(tokenized_docs, show_progress=self.config.show_progress_bar)
        return stats

    def _get_token_score(self, stats: TokenTFIDFStats, doc_idx: int, token_id: int) -> float:
        """Get the score for a token (IDF or TF-IDF)."""
        if self.config.use_tfidf:
            return stats.get_tfidf(doc_idx, token_id)
        else:
            return stats.get_idf(token_id)

    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply IDF-based pooling to embeddings.

        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain "input_ids" as a list of tensors.

        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pooled_embeddings, updated_artifacts)
        """
        if "input_ids" not in artifacts:
            raise ValueError(
                "IDFPoolingStrategy requires 'input_ids' artifact. "
                "Ensure input_ids are provided when encoding."
            )

        input_ids = artifacts["input_ids"]
        if not isinstance(input_ids, list) or len(input_ids) != len(embeddings):
            raise ValueError("input_ids must be a list matching embeddings length")

        # Compute TF-IDF statistics
        if "tfidf_stats" in artifacts:
            stats = artifacts["tfidf_stats"]
        else:
            stats = self._compute_tfidf_stats(input_ids)

        pooled_embeddings = []
        doc_mappings = []

        iterator = tqdm(
            enumerate(zip(embeddings, input_ids)),
            desc=f"IDF pooling (keep_ratio={self.config.keep_ratio})",
            total=len(embeddings),
            disable=not self.config.show_progress_bar,
        )

        for doc_idx, (doc_emb, doc_input_ids) in iterator:
            device = doc_emb.device
            n_tokens, dim = doc_emb.shape
            doc_tokens = doc_input_ids.cpu().tolist()

            if n_tokens == 0:
                pooled_embeddings.append(doc_emb)
                doc_mappings.append({"protected": 0, "cluster_labels": [], "num_clusters": 0})
                continue

            protected = min(self.config.protected_tokens, n_tokens)
            n_non_protected = max(n_tokens - protected, 0)

            if n_non_protected <= 0:
                pooled_embeddings.append(doc_emb)
                doc_mappings.append({"protected": protected, "cluster_labels": [], "num_clusters": 0})
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

            # Compute IDF scores for non-protected tokens
            token_scores: list[tuple[int, float]] = []  # (local_idx, score)
            for local_idx in range(n_non_protected):
                token_id = doc_tokens[protected + local_idx]

                # Skip ignored tokens - give them minimum score
                if self.config.ignore_token_ids and token_id in self.config.ignore_token_ids:
                    token_scores.append((local_idx, float('-inf')))
                else:
                    score = self._get_token_score(stats, doc_idx, token_id)
                    token_scores.append((local_idx, score))

            # Select anchors: tokens with highest IDF scores
            sorted_by_score = sorted(token_scores, key=lambda x: x[1], reverse=True)
            num_anchors = min(num_clusters, n_non_protected)
            anchor_local_indices = [idx for idx, _ in sorted_by_score[:num_anchors]]
            anchor_local_indices.sort()  # Sort by position to maintain document order

            anchor_local_idx = torch.tensor(anchor_local_indices, device=device)
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
            for cid in range(num_anchors):
                mask = cluster_labels_tensor == cid
                if not mask.any():
                    continue
                members = nonprot_emb[mask]
                pooled_vec = members.mean(dim=0)
                pooled_cluster_vectors.append(pooled_vec)

            num_clusters_effective = len(pooled_cluster_vectors)
            if num_clusters_effective == 0:
                pooled_embeddings.append(doc_emb)
                doc_mappings.append({"protected": protected, "cluster_labels": cluster_labels, "num_clusters": 0})
                continue

            # Combine protected + pooled
            pooled_cluster_stacked = torch.stack(pooled_cluster_vectors, dim=0)
            final_emb = torch.cat([prot_emb, pooled_cluster_stacked], dim=0)
            pooled_embeddings.append(final_emb)

            doc_mappings.append({
                "protected": protected,
                "cluster_labels": cluster_labels,
                "num_clusters": num_clusters_effective,
            })

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
            if artifact_name == "input_ids":
                # Pool input_ids: keep protected tokens + representative token per cluster
                pooled_input_ids = []
                for doc_idx, doc_input_ids in enumerate(artifact_value):
                    mapping = doc_mappings[doc_idx]
                    protected = mapping["protected"]
                    cluster_labels = mapping["cluster_labels"]
                    num_clusters = mapping["num_clusters"]

                    if num_clusters == 0:
                        pooled_input_ids.append(doc_input_ids)
                        continue

                    prot_tokens = doc_input_ids[:protected]

                    # For each cluster, pick the first member's token as representative
                    pooled_cluster_tokens = []
                    for cid in range(num_clusters):
                        member_indices = [i for i, lab in enumerate(cluster_labels) if lab == cid]
                        if not member_indices:
                            continue
                        first_local = member_indices[0]
                        orig_idx = protected + first_local
                        if orig_idx < len(doc_input_ids):
                            pooled_cluster_tokens.append(doc_input_ids[orig_idx])

                    if pooled_cluster_tokens:
                        pooled_cluster_tensor = torch.stack(pooled_cluster_tokens)
                        final_tokens = torch.cat([prot_tokens, pooled_cluster_tensor])
                    else:
                        final_tokens = prot_tokens

                    pooled_input_ids.append(final_tokens)

                updated_artifacts[artifact_name] = pooled_input_ids

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

                    is_tensor = isinstance(artifact_tokens, torch.Tensor)
                    n_tokens_art = artifact_tokens.size(0) if is_tensor else len(artifact_tokens)
                    prot_tokens = artifact_tokens[:protected]

                    pooled_cluster_tokens = []
                    for cid in range(num_clusters):
                        member_indices = [i for i, lab in enumerate(cluster_labels) if lab == cid]
                        if not member_indices:
                            continue
                        first_local = member_indices[0]
                        orig_idx = protected + first_local
                        if orig_idx < n_tokens_art:
                            pooled_cluster_tokens.append(artifact_tokens[orig_idx])

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

