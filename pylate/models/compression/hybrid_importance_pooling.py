from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import (
    CompressionArtifacts,
    CompressionStrategy,
    CompressionStrategyConfigBase,
)
from ..utils import TokenTFIDFStats

try:
    import fastkmeans
except ImportError:
    fastkmeans = None

try:
    from scipy.cluster import hierarchy
except ImportError:
    hierarchy = None


@dataclass
class HybridPoolingConfig(CompressionStrategyConfigBase):
    """
    Config for hybrid importance + clustering pooling.

    Parameters
    ----------
    pool_factor:
        Controls the number of clusters: num_clusters = max(K // pool_factor, 1),
        where K is the number of non-protected tokens.
    keep_ratio:
        Fraction of non-protected tokens to use as anchors (0 < keep_ratio <= 1).
        num_anchors = max(int(keep_ratio * K), 1).
    protected_tokens:
        Number of leading tokens in each document that are never pooled
        (they are copied as-is at the front of the pooled sequence).
    min_tokens:
        Minimum total number of vectors per document after pooling
        (including protected tokens).
    clustering_method:
        "hierarchical" or "spherical".
    show_progress_bar:
        Whether to show a tqdm progress bar.
    use_norm:
        Whether to use L2 norm of embeddings as part of the importance score.
    use_idf:
        Whether to use IDF scores from artifacts["idf"] if available.
        Each artifacts["idf"][doc_idx] should be a tensor or list of length doc_len.
    use_token_weights:
        Whether to use arbitrary per-token weights from artifacts["token_weights"].
    norm_weight, idf_weight, token_weights_weight:
        Coefficients for combining the three importance signals.
    """

    pool_factor: int = 4
    keep_ratio: float = 0.5
    protected_tokens: int = 1
    min_tokens: int = 8
    clustering_method: str = "hierarchical"
    show_progress_bar: bool = False

    use_norm: bool = True
    use_idf: bool = False
    use_token_weights: bool = False
    norm_weight: float = 1.0
    idf_weight: float = 1.0
    token_weights_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.pool_factor <= 0:
            raise ValueError("pool_factor must be a positive integer.")
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1].")
        if self.protected_tokens < 0:
            raise ValueError("protected_tokens must be >= 0.")
        if self.min_tokens <= 0:
            raise ValueError("min_tokens must be > 0.")
        if self.clustering_method not in ("hierarchical", "spherical"):
            raise ValueError("clustering_method must be 'hierarchical' or 'spherical'.")

    def serialize(self) -> dict:
        return {
            "pool_factor": self.pool_factor,
            "keep_ratio": self.keep_ratio,
            "protected_tokens": self.protected_tokens,
            "min_tokens": self.min_tokens,
            "clustering_method": self.clustering_method,
            "show_progress_bar": self.show_progress_bar,
            "use_norm": self.use_norm,
            "use_idf": self.use_idf,
            "use_token_weights": self.use_token_weights,
            "norm_weight": self.norm_weight,
            "idf_weight": self.idf_weight,
            "token_weights_weight": self.token_weights_weight,
        }

    @property
    def strategy_type(self) -> str:
        has_norm = 'Norm' if self.use_norm and self.norm_weight > 0.0 else ''
        has_idf = 'IDF' if self.use_idf and self.idf_weight > 0.0 else ''
        has_token_weights = 'W' if self.use_token_weights and self.token_weights_weight > 0.0 else ''
        return f"hybrid_imp+clust_pooling_{has_norm}{has_idf}{has_token_weights}"


class HybridImportanceClusteringPoolingStrategy(CompressionStrategy):
    """
    Hybrid importance + clustering pooling for multi-vector retrieval.

    For each document:

      1. Protect the first `protected_tokens` embeddings (copied as-is).
      2. On the remaining K tokens:
         - Run clustering (hierarchical or spherical) into C clusters,
           where C = max(K // pool_factor, 1).
         - Compute an importance score for each token:
             score_i = w_norm * ||e_i||_2
                     + w_idf * idf_i
                     + w_tw  * token_weight_i
           using whatever components are enabled & available.
         - Select A anchors by importance:
             A = max(int(keep_ratio * K), 1), A <= K.
      3. For each cluster:
         - Let members be the tokens in this cluster (local indices 0..K-1).
         - anchors_in_cluster = members ∩ anchors.
         - If anchors_in_cluster is non-empty:
             * Use those anchors as "dynamic centroids".
             * Assign every member in the cluster (including anchors) to its
               nearest anchor (cosine similarity).
             * For each anchor, average embeddings of its assigned members
               into one pooled vector.
           Else:
             * Fallback to a single cluster centroid = mean of member embeddings.
      4. Final doc representation:
         [protected tokens] + [all pooled vectors from all clusters].

    For artifacts, each pooled vector chooses a representative original token
    (usually the anchor itself or the first member in the cluster) and copies
    that token's artifact (e.g., token id, span) as the pooled artifact.
    """

    required_artifacts: List[str] = []

    def __init__(self, config: HybridPoolingConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return (
            f"hybrid-imp-pool_{self.config.clustering_method}"
            f"_pf-{self.config.pool_factor}"
            f"_kr-{self.config.keep_ratio:.2f}"
            f"_p-{self.config.protected_tokens}"
        )

    @property
    def strategy_type(self) -> str:
        return self.config.strategy_type

    def serialize(self) -> dict:
        return {
            "type": self.strategy_type,
            "config": self.config.serialize(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HybridImportanceClusteringPoolingStrategy":
        cfg = data.get("config", {}) or {}
        config = HybridPoolingConfig(
            pool_factor=cfg.get("pool_factor", 4),
            keep_ratio=cfg.get("keep_ratio", 0.5),
            protected_tokens=cfg.get("protected_tokens", 1),
            min_tokens=cfg.get("min_tokens", 8),
            clustering_method=cfg.get("clustering_method", "hierarchical"),
            show_progress_bar=cfg.get("show_progress_bar", False),
            use_norm=cfg.get("use_norm", True),
            use_idf=cfg.get("use_idf", False),
            use_token_weights=cfg.get("use_token_weights", False),
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

    def _cluster_nonprotected(
        self,
        nonprot_emb: torch.Tensor,
    ) -> List[int]:
        """
        Cluster non-protected embeddings and return a list of cluster labels
        (local indices 0..K-1 -> label in [0..C-1]).
        """
        K, dim = nonprot_emb.shape

        if K == 0:
            return []

        num_clusters = max(K // self.config.pool_factor, 1)

        if self.config.clustering_method == "hierarchical":
            if hierarchy is None:
                raise ImportError(
                    "scipy.cluster.hierarchy is required for hierarchical clustering."
                )

            # Cosine distance matrix
            norms = torch.norm(nonprot_emb, dim=-1, keepdim=True)
            sims = (nonprot_emb / (norms + 1e-8)) @ (nonprot_emb / (norms + 1e-8)).t()
            dist = 1.0 - sims.cpu().numpy()

            Z = hierarchy.linkage(dist, method="ward")
            cluster_labels = hierarchy.fcluster(
                Z, t=num_clusters, criterion="maxclust"
            )  # 1..num_clusters
            cluster_labels = [int(c - 1) for c in cluster_labels]
            return cluster_labels

        if self.config.clustering_method == "spherical":
            if fastkmeans is None:
                raise ImportError(
                    "fastkmeans is required for spherical clustering. "
                    "Install it with: pip install fastkmeans"
                )

            emb_norm = F.normalize(nonprot_emb, p=2, dim=-1)
            emb_np = emb_norm.cpu().float().numpy()
            embedding_dim = emb_np.shape[1]

            kmeans = fastkmeans.FastKMeans(
                embedding_dim,
                num_clusters,
                niter=10,
                gpu=False,
                verbose=False,
                seed=42,
            )
            kmeans.train(emb_np)
            labels_np = kmeans.predict(emb_np)
            labels = labels_np.tolist()
            return [int(c) for c in labels]

        raise ValueError(
            f"Unknown clustering method: {self.config.clustering_method}"
        )

    def compress(
        self,
        embeddings: List[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> Tuple[List[torch.Tensor], CompressionArtifacts]:
        """
        Apply hybrid importance + clustering pooling and update artifacts.

        Returns
        -------
        (pooled_embeddings, updated_artifacts)
        """

        if self.config.pool_factor == 1 and self.config.keep_ratio >= 0.999:
            return embeddings, artifacts

        idf_docs = artifacts.get("idf", None)
        token_weight_docs = artifacts.get("token_weights", None)

        # Compute IDF if needed and not provided
        if self.config.use_idf and idf_docs is None:
            input_ids = artifacts.get("input_ids", None)
            if input_ids is not None:
                idf_docs = self._compute_document_idf(input_ids, show_progress=self.config.show_progress_bar)

        pooled_embeddings: List[torch.Tensor] = []
        rep_indices_per_doc: List[List[int]] = []

        iterator = tqdm(
            range(len(embeddings)),
            desc=(
                f"Hybrid imp+cluster pooling "
                f"({self.config.clustering_method}, pf={self.config.pool_factor}, "
                f"kr={self.config.keep_ratio:.2f})"
            ),
            disable=not self.config.show_progress_bar,
            leave=False,
        )

        for doc_idx in iterator:
            doc_emb = embeddings[doc_idx]
            device = doc_emb.device
            n_tokens, dim = doc_emb.shape

            if n_tokens == 0:
                pooled_embeddings.append(doc_emb)
                rep_indices_per_doc.append([])
                continue

            protected = min(self.config.protected_tokens, n_tokens)
            n_nonprot = max(n_tokens - protected, 0)

            protected_emb = doc_emb[:protected]
            rep_indices: List[int] = list(range(protected))

            if n_nonprot <= 0:
                pooled_embeddings.append(protected_emb)
                rep_indices_per_doc.append(rep_indices)
                continue

            K = n_nonprot
            target_total = max(
                self.config.min_tokens,
                protected + int(self.config.keep_ratio * K),
            )
            target_total = min(target_total, n_tokens)

            if target_total >= n_tokens:
                pooled_embeddings.append(doc_emb)
                rep_indices_per_doc.append(list(range(n_tokens)))
                continue

            num_anchors = max(int(self.config.keep_ratio * K), 1)
            num_anchors = min(num_anchors, K)

            nonprot_emb = doc_emb[protected:, :]
            cluster_labels = self._cluster_nonprotected(nonprot_emb)
            if len(cluster_labels) != K:
                raise RuntimeError("Cluster labels length mismatch.")

            C = max(set(cluster_labels)) + 1

            if idf_docs is not None and self.config.use_idf:
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

            if token_weight_docs is not None and self.config.use_token_weights:
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
            )

            _, anchor_local_idx = torch.topk(
                scores, k=num_anchors, largest=True, sorted=False
            )
            anchors_set = set(anchor_local_idx.tolist())

            nonprot_norm = F.normalize(nonprot_emb, p=2, dim=-1)

            pooled_cluster_vecs: List[torch.Tensor] = []
            pooled_cluster_rep_indices: List[int] = []

            members_per_cluster: List[List[int]] = [[] for _ in range(C)]
            for i_local, lbl in enumerate(cluster_labels):
                members_per_cluster[lbl].append(i_local)

            for c in range(C):
                members = members_per_cluster[c]
                if not members:
                    continue

                anchors_in_cluster = [i for i in members if i in anchors_set]

                if anchors_in_cluster:
                    anchor_emb = nonprot_emb[anchors_in_cluster]
                    anchor_norm = F.normalize(anchor_emb, p=2, dim=-1)

                    members_emb = nonprot_norm[members]
                    sims = members_emb @ anchor_norm.t()
                    assign = sims.argmax(dim=-1)

                    for a_idx_local_cluster, anchor_local in enumerate(anchors_in_cluster):
                        member_positions = [
                            j for j, a in zip(members, assign.tolist()) if a == a_idx_local_cluster
                        ]
                        if not member_positions:
                            continue
                        emb_to_pool = nonprot_emb[member_positions]
                        pooled_vec = emb_to_pool.mean(dim=0)
                        pooled_cluster_vecs.append(pooled_vec)

                        rep_global_idx = protected + anchor_local
                        pooled_cluster_rep_indices.append(rep_global_idx)
                else:
                    emb_to_pool = nonprot_emb[members]
                    pooled_vec = emb_to_pool.mean(dim=0)
                    pooled_cluster_vecs.append(pooled_vec)
                    rep_global_idx = protected + members[0]
                    pooled_cluster_rep_indices.append(rep_global_idx)

            order = sorted(
                range(len(pooled_cluster_vecs)),
                key=lambda i: pooled_cluster_rep_indices[i],
            )
            pooled_cluster_vecs_sorted = [pooled_cluster_vecs[i] for i in order]
            rep_indices_sorted = [pooled_cluster_rep_indices[i] for i in order]

            if pooled_cluster_vecs_sorted:
                pooled_clusters_tensor = torch.stack(pooled_cluster_vecs_sorted, dim=0)
                pooled_doc = torch.cat([protected_emb, pooled_clusters_tensor], dim=0)
            else:
                pooled_doc = protected_emb
                rep_indices_sorted = []

            # Enforce min_tokens by falling back to no pooling if we pooled too much
            if pooled_doc.shape[0] < self.config.min_tokens and n_tokens >= self.config.min_tokens:
                pooled_doc = doc_emb
                rep_indices = list(range(n_tokens))
            else:
                rep_indices.extend(rep_indices_sorted)

            pooled_embeddings.append(pooled_doc)
            rep_indices_per_doc.append(rep_indices)

        updated_artifacts: CompressionArtifacts = {}

        for name, value in artifacts.items():
            if not isinstance(value, list):
                updated_artifacts[name] = value
                continue

            pooled_artifacts_for_name: List[Any] = []

            for doc_idx, artifact_tokens in enumerate(value):
                rep_indices = rep_indices_per_doc[doc_idx]

                if isinstance(artifact_tokens, torch.Tensor):
                    length = artifact_tokens.size(0)
                    clipped = [i for i in rep_indices if i < length]
                    if not clipped:
                        pooled_artifacts_for_name.append(artifact_tokens)
                        continue
                    rep_idx_tensor = torch.tensor(
                        clipped,
                        device=artifact_tokens.device,
                        dtype=torch.long,
                    )
                    new_tokens = artifact_tokens[rep_idx_tensor]
                else:
                    tokens_list = list(artifact_tokens)
                    length = len(tokens_list)
                    clipped = [i for i in rep_indices if i < length]
                    if not clipped:
                        pooled_artifacts_for_name.append(tokens_list)
                        continue
                    new_tokens = [tokens_list[i] for i in clipped]

                pooled_artifacts_for_name.append(new_tokens)

            updated_artifacts[name] = pooled_artifacts_for_name

        return pooled_embeddings, updated_artifacts
