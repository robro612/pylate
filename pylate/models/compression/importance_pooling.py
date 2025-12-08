from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from .base import CompressionStrategy, CompressionStrategyConfigBase, CompressionArtifacts
from ..utils import TokenTFIDFStats

@dataclass
class ImportancePoolingConfig(CompressionStrategyConfigBase):
    """
    Configuration for training-free importance-based token pooling.

    Parameters
    ----------
    keep_ratio
        Fraction of non-protected tokens to represent with distinct pooled vectors.
        (0 < keep_ratio <= 1). The actual number of output vectors will be:
            protected_tokens + max(min_tokens - protected_tokens,
                                   int(keep_ratio * (doc_len - protected_tokens)))
    protected_tokens
        Number of tokens at the start of each document that are never pooled.
        (They stay as individual vectors; typically [CLS]/specials.)
    min_tokens
        Minimum total number of vectors per document after pooling
        (including protected tokens).
    use_norm
        Whether to use the L2 norm of embeddings as part of the importance score.
    use_idf
        Whether to use IDF scores from artifacts["idf"] if available.
        Expected shape: List[Tensor] or List[List[float]] matching documents.
    use_token_weights
        Whether to use arbitrary per-token weights from artifacts["token_weights"].
        This can encode heuristics like "is_content", "not stopword", etc.
    norm_weight
        Coefficient for the L2 norm component in the importance score.
    idf_weight
        Coefficient for the IDF component in the importance score.
    token_weights_weight
        Coefficient for the token_weight component in the importance score.
    """
    keep_ratio: float = 0.5
    protected_tokens: int = 1
    min_tokens: int = 8
    use_norm: bool = True
    use_idf: bool = True
    use_token_weights: bool = True
    norm_weight: float = 1.0
    idf_weight: float = 1.0
    token_weights_weight: float = 1.0

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
            "use_norm": self.use_norm,
            "use_idf": self.use_idf,
            "use_token_weights": self.use_token_weights,
            "norm_weight": self.norm_weight,
            "idf_weight": self.idf_weight,
            "token_weights_weight": self.token_weights_weight,
        }

    @property
    def strategy_type(self) -> str:
        parts = []
        if self.use_norm and self.norm_weight > 0.0:
            parts.append("Norm")
        if self.use_idf and self.idf_weight > 0.0:
            parts.append("IDF")
        if self.use_token_weights and self.token_weights_weight > 0.0:
            parts.append("W")
        suffix = "".join(parts)
        return f"importance_pooling_{suffix}" if suffix else "importance_pooling"


class ImportancePoolingStrategy(CompressionStrategy):
    """
    Training-free importance-based pooling for multi-vector retrieval.

    Conceptually:

      1. Protect the first `protected_tokens` in each document.
      2. For the remaining tokens, compute an importance score:
           score_i = w_norm * ||e_i||_2
                   + w_idf * idf_i
                   + w_tw  * token_weight_i
         using whatever components are enabled & available.
      3. Select C "anchor" tokens with highest importance (C depends on keep_ratio/min_tokens).
      4. Assign every non-protected token to its nearest anchor (cosine sim).
      5. Pool each cluster by averaging embeddings -> one pooled vector per anchor.
      6. For artifacts, keep protected tokens as-is and, for each cluster,
         reuse the artifact of the anchor token as the representative.

    This reduces the number of vectors per document while preserving
    important semantic directions, and it's fully training-free.
    """

    required_artifacts: list[str] = []  # "idf", "token_weights" are optional hints

    def __init__(self, config: ImportancePoolingConfig) -> None:
        """
        Initialize the importance pooling strategy.

        Parameters
        ----------
        config
            Importance pooling configuration
        """
        self.config = config

    @property
    def name(self) -> str:
        return (
            f"importance-pool_r-{self.config.keep_ratio:.2f}_"
            f"p-{self.config.protected_tokens}_"
            f"min-{self.config.min_tokens}"
        )

    @property
    def strategy_type(self) -> str:
        return "importance_pooling"

    def serialize(self) -> dict:
        """
        Serialize this strategy to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "type": self.strategy_type,
            "config": self.config.serialize(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImportancePoolingStrategy":
        """
        Create a strategy instance from a serialized dictionary.

        Parameters
        ----------
        data
            Dictionary containing serialized strategy data

        Returns
        -------
        ImportancePoolingStrategy
            Deserialized strategy instance
        """
        cfg = data.get("config", {}) or {}
        config = ImportancePoolingConfig(
            keep_ratio=cfg.get("keep_ratio", 0.5),
            protected_tokens=cfg.get("protected_tokens", 1),
            min_tokens=cfg.get("min_tokens", 8),
            use_norm=cfg.get("use_norm", True),
            use_idf=cfg.get("use_idf", True),
            use_token_weights=cfg.get("use_token_weights", True),
            norm_weight=cfg.get("norm_weight", 1.0),
            idf_weight=cfg.get("idf_weight", 1.0),
            token_weights_weight=cfg.get("token_weights_weight", 1.0),
        )
        return cls(config)

    def _compute_document_idf(
        self,
        input_ids: list[torch.Tensor],
        show_progress: bool = False,
    ) -> list[torch.Tensor]:
        """
        Compute document-wise IDF scores for each token.

        Parameters
        ----------
        input_ids
            List of input_id tensors (one per document)
        show_progress
            Whether to show progress bar

        Returns
        -------
        list[torch.Tensor]
            List of IDF score tensors (one per document), same shape as input_ids
        """
        # Convert tensors to lists for TokenTFIDFStats
        tokenized_docs = [doc_input_ids.cpu().tolist() for doc_input_ids in input_ids]

        # Compute TF-IDF statistics
        stats = TokenTFIDFStats(num_docs=len(tokenized_docs))
        stats.fit(tokenized_docs, show_progress=show_progress)

        # Extract IDF scores for each document
        idf_scores = []
        for doc_idx, doc_tokens in enumerate(tokenized_docs):
            doc_idf = torch.tensor(
                [stats.get_idf(token_id) for token_id in doc_tokens],
                dtype=torch.float32,
            )
            idf_scores.append(doc_idf)

        return idf_scores

    def _get_artifact_vector(
        self,
        artifact_value_for_doc: Any,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
        default_val: float = 0.0,
    ) -> torch.Tensor:
        """
        Normalize a per-token artifact (e.g., idf or token_weights) to a
        length-`length` tensor on the given device, with the given dtype.

        Parameters
        ----------
        artifact_value_for_doc
            The artifact value for a single document (can be Tensor, list, or None)
        length
            Expected length of the artifact vector
        device
            Target device for the tensor
        dtype
            Target dtype for the tensor
        default_val
            Default value to use if artifact is None or needs padding

        Returns
        -------
        torch.Tensor
            Normalized artifact vector of shape (length,)
        """
        if artifact_value_for_doc is None:
            return torch.full((length,), float(default_val), device=device, dtype=dtype)

        if isinstance(artifact_value_for_doc, torch.Tensor):
            vec = artifact_value_for_doc.to(device=device, dtype=dtype)
        else:
            vec = torch.tensor(artifact_value_for_doc, device=device, dtype=dtype)

        if vec.numel() < length:
            padding = torch.full(
                (length - vec.numel(),),
                float(default_val),
                device=device,
                dtype=dtype,
            )
            vec = torch.cat([vec, padding], dim=0)
        return vec[:length]

    def _compute_importance_scores(
        self,
        doc_embeddings: torch.Tensor,
        idf_vec: Optional[torch.Tensor],
        token_weight_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute importance scores for all tokens in a single document,
        given embeddings and optional per-token signals.

        Parameters
        ----------
        doc_embeddings
            Embeddings for a single document, shape (n_tokens, dim)
        idf_vec
            Optional IDF scores for each token, shape (n_tokens,)
        token_weight_vec
            Optional token weights for each token, shape (n_tokens,)

        Returns
        -------
        torch.Tensor
            Importance scores for each token, shape (n_tokens,)
        """
        device = doc_embeddings.device
        n_tokens = doc_embeddings.size(0)
        scores = torch.zeros(n_tokens, device=device)

        if self.config.use_norm:
            norms = torch.norm(doc_embeddings, dim=-1)
            scores = scores + self.config.norm_weight * norms

        if self.config.use_idf and idf_vec is not None:
            scores = scores + self.config.idf_weight * idf_vec

        if self.config.use_token_weights and token_weight_vec is not None:
            scores = scores + self.config.token_weights_weight * token_weight_vec

        return scores

    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply importance-based pooling to embeddings and update artifacts.

        Parameters
        ----------
        embeddings
            List[Tensor], one tensor per document, shape [doc_len, dim].
        artifacts
            Dict of artifacts. Shape-matched artifacts (lists of per-doc sequences)
            are pooled to keep a 1:1 mapping with pooled embeddings.
            Metadata artifacts are passed through unchanged.

        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pooled_embeddings, updated_artifacts)
        """
        # Optional global per-token artifacts
        idf_docs = artifacts.get("idf", None)
        token_weight_docs = artifacts.get("token_weights", None)

        # Compute IDF if needed and not provided
        if self.config.use_idf and idf_docs is None:
            input_ids = artifacts.get("input_ids", None)
            if input_ids is not None:
                idf_docs = self._compute_document_idf(input_ids, show_progress=False)

        pooled_embeddings: list[torch.Tensor] = []

        # For artifact mapping we need, per doc:
        #   protected_count, cluster_labels (for non-protected), num_clusters
        doc_mappings: list[dict[str, Any]] = []

        for doc_idx, doc_emb in enumerate(embeddings):
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

            # Target number of total vectors after pooling
            target_total = max(
                self.config.min_tokens,
                protected + int(self.config.keep_ratio * n_non_protected),
            )
            target_total = min(target_total, n_tokens)
            num_clusters = max(target_total - protected, 1)

            # If num_clusters >= n_non_protected, it means no real pooling needed
            if num_clusters >= n_non_protected:
                pooled_embeddings.append(doc_emb)
                # cluster_labels: identity assignment for completeness
                cluster_labels = list(range(n_non_protected))
                doc_mappings.append(
                    {
                        "protected": protected,
                        "cluster_labels": cluster_labels,
                        "num_clusters": n_non_protected,
                    }
                )
                continue

            # Slice non-protected embeddings
            nonprot_emb = doc_emb[protected:, :]  # [K, dim]
            K = nonprot_emb.size(0)

            # Build per-token artifacts for scoring (optional)
            if idf_docs is not None:
                idf_full = self._get_artifact_vector(
                    idf_docs[doc_idx],
                    length=n_tokens,
                    device=device,
                    dtype=torch.float32,
                    default_val=0.0,
                )
                idf_vec = idf_full[protected:]
            else:
                idf_vec = None

            if token_weight_docs is not None:
                tw_full = self._get_artifact_vector(
                    token_weight_docs[doc_idx],
                    length=n_tokens,
                    device=device,
                    dtype=torch.float32,
                    default_val=0.0,
                )
                token_weight_vec = tw_full[protected:]
            else:
                token_weight_vec = None

            scores = self._compute_importance_scores(
                doc_embeddings=nonprot_emb,
                idf_vec=idf_vec,
                token_weight_vec=token_weight_vec,
            )  # [K]

            # Select num_clusters most important tokens as anchors
            _, anchor_local_idx = torch.topk(
                scores, k=num_clusters, largest=True, sorted=False
            )
            # Sort anchors by position to keep some document-order consistency
            anchor_local_idx, _ = torch.sort(anchor_local_idx)  # [C]

            anchor_emb = nonprot_emb[anchor_local_idx]  # [C, dim]

            # Assign each token to nearest anchor (cosine similarity)
            import torch.nn.functional as F
            nonprot_norm = F.normalize(nonprot_emb, p=2, dim=-1)
            anchor_norm = F.normalize(anchor_emb, p=2, dim=-1)
            sims = nonprot_norm @ anchor_norm.t()  # [K, C]
            cluster_labels_tensor = sims.argmax(dim=-1)  # [K], values in [0, C-1]
            cluster_labels = cluster_labels_tensor.cpu().tolist()

            # Pool embeddings for each cluster
            pooled_cluster_vectors: list[torch.Tensor] = []
            for cid in range(num_clusters):
                mask = cluster_labels_tensor == cid
                if not mask.any():
                    # Should be rare; just skip this cluster
                    continue
                members = nonprot_emb[mask]  # [num_members, dim]
                pooled_vec = members.mean(dim=0)
                pooled_cluster_vectors.append(pooled_vec)

            # If some clusters ended up empty (very rare), adjust num_clusters
            num_clusters_effective = len(pooled_cluster_vectors)
            if num_clusters_effective == 0:
                # Fallback: no pooling
                pooled_embeddings.append(doc_emb)
                doc_mappings.append(
                    {
                        "protected": protected,
                        "cluster_labels": cluster_labels,
                        "num_clusters": 0,
                    }
                )
                continue

            # Build final pooled embedding matrix: [protected tokens; pooled clusters]
            protected_emb = doc_emb[:protected]
            pooled_doc = torch.cat(
                [protected_emb, torch.stack(pooled_cluster_vectors, dim=0)],
                dim=0,
            )
            pooled_embeddings.append(pooled_doc)

            doc_mappings.append(
                {
                    "protected": protected,
                    "cluster_labels": cluster_labels,  # length K
                    "num_clusters": num_clusters_effective,
                }
            )

        # Update artifacts using doc_mappings
        updated_artifacts: CompressionArtifacts = {}

        for name, value in artifacts.items():
            # Metadata: carried as-is
            if not isinstance(value, list):
                updated_artifacts[name] = value
                continue

            pooled_artifacts_for_name: list[Any] = []

            for doc_idx, artifact_tokens in enumerate(value):
                mapping = doc_mappings[doc_idx]
                protected = mapping["protected"]
                cluster_labels = mapping["cluster_labels"]
                num_clusters = mapping["num_clusters"]

                # If no pooling happened for this doc or no clustering info, just pass through
                if num_clusters == 0 or len(cluster_labels) == 0:
                    pooled_artifacts_for_name.append(artifact_tokens)
                    continue

                # Normalize artifact_tokens to something indexable
                is_tensor = isinstance(artifact_tokens, torch.Tensor)
                if is_tensor:
                    tokens_arr = artifact_tokens
                    n_tokens_art = tokens_arr.size(0)
                else:
                    tokens_list = list(artifact_tokens)
                    n_tokens_art = len(tokens_list)

                # Protected part: indices [0, protected)
                if is_tensor:
                    protected_part = (
                        tokens_arr[:protected] if protected <= n_tokens_art else tokens_arr
                    )
                else:
                    protected_part = tokens_list[:protected]

                # For each cluster, pick a representative token (the first member)
                pooled_cluster_tokens: list[Any] = []
                K = len(cluster_labels)  # non-protected token count

                for cid in range(num_clusters):
                    # find first non-protected index whose cluster_label == cid
                    member_local_indices = [
                        i for i, lab in enumerate(cluster_labels) if lab == cid
                    ]
                    if not member_local_indices:
                        continue
                    first_local = member_local_indices[0]
                    orig_idx = protected + first_local  # index in full doc

                    if orig_idx >= n_tokens_art:
                        continue

                    if is_tensor:
                        pooled_cluster_tokens.append(tokens_arr[orig_idx])
                    else:
                        pooled_cluster_tokens.append(tokens_list[orig_idx])

                # Build pooled artifact sequence
                if is_tensor:
                    # protected_part may be shorter if artifact_tokens is shorter
                    pieces = []
                    if protected_part.numel() > 0:
                        pieces.append(protected_part)
                    if len(pooled_cluster_tokens) > 0:
                        pooled_cluster_tensor = torch.stack(
                            pooled_cluster_tokens, dim=0
                        )
                        pieces.append(pooled_cluster_tensor)
                    if pieces:
                        new_tokens = torch.cat(pieces, dim=0)
                    else:
                        new_tokens = tokens_arr  # Fallback
                else:
                    new_tokens = list(protected_part) + list(pooled_cluster_tokens)

                pooled_artifacts_for_name.append(new_tokens)

            updated_artifacts[name] = pooled_artifacts_for_name

        return pooled_embeddings, updated_artifacts
