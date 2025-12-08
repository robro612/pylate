from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from tqdm.autonotebook import tqdm

from .base import CompressionStrategy, CompressionStrategyConfigBase, CompressionArtifacts
from ..utils import TokenTFIDFStats

@dataclass
class ImportancePruningConfig(CompressionStrategyConfigBase):
    """
    Configuration for training-free importance-based token pruning.

    Parameters
    ----------
    keep_ratio
        Fraction of non-protected tokens to keep (0 < keep_ratio <= 1).
    protected_tokens
        Number of tokens at the start of each document that are never pruned.
        Typically used to keep special tokens / CLS / BOS.
    min_tokens
        Minimum total number of tokens per document after pruning
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
        return f"importance_pruning_{suffix}" if suffix else "importance_pruning"

class ImportancePruningStrategy(CompressionStrategy):
    """
    Training-free importance-based token pruning for multi-vector retrieval.

    This strategy:
      1. Protects the first `protected_tokens` in each document.
      2. For the remaining tokens, computes an importance score:
         score_i = w_norm * ||e_i||_2
                 + w_idf * idf_i
                 + w_tw  * token_weight_i
         using whatever components are enabled & available.
      3. Keeps top-K tokens by score, where K is derived from `keep_ratio`
         and `min_tokens`.
      4. Updates all shape-matched artifacts so that they remain aligned
         with the pruned token positions.

    It is fully training-free and designed to be a stronger baseline than
    naive "cluster and average everything" in many settings.
    """

    required_artifacts: list[str] = []  # "idf", "token_weights" are optional hints

    def __init__(self, config: ImportancePruningConfig) -> None:
        """
        Initialize the importance pruning strategy.

        Parameters
        ----------
        config
            Importance pruning configuration
        """
        self.config = config

    @property
    def name(self) -> str:
        return (
            f"importance-prune_r-{self.config.keep_ratio:.2f}_"
            f"p-{self.config.protected_tokens}_"
            f"min-{self.config.min_tokens}"
        )

    @property
    def strategy_type(self) -> str:
        return self.config.strategy_type

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
    def from_dict(cls, data: dict) -> "ImportancePruningStrategy":
        """
        Create a strategy instance from a serialized dictionary.

        Parameters
        ----------
        data
            Dictionary containing serialized strategy data

        Returns
        -------
        ImportancePruningStrategy
            Deserialized strategy instance
        """
        cfg = data.get("config", {}) or {}
        config = ImportancePruningConfig(
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
            vec = artifact_value_for_doc
        else:
            # Assume list or other iterable
            vec = torch.tensor(artifact_value_for_doc, device=device, dtype=dtype)

        if vec.numel() < length:
            # Pad at the end with default_val
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
            Embedding tensor for one document, shape (num_tokens, embedding_dim)
        idf_vec
            Optional IDF scores for each token, shape (num_tokens,)
        token_weight_vec
            Optional token weights for each token, shape (num_tokens,)

        Returns
        -------
        torch.Tensor
            Importance scores for each token, shape (num_tokens,)
        """
        device = doc_embeddings.device
        n_tokens = doc_embeddings.size(0)
        scores = torch.zeros(n_tokens, device=device)

        if self.config.use_norm:
            # L2 norm of each token vector
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
        Apply importance-based pruning to embeddings and update artifacts.

        Parameters
        ----------
        embeddings
            List[Tensor], one tensor per document, shape [doc_len, dim].
        artifacts
            Dict of artifacts. Shape-matched artifacts (lists of per-doc
            sequences) are pruned to keep a 1:1 mapping with embeddings.
            Metadata artifacts are passed through unchanged.

        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (compressed_embeddings, updated_artifacts)
        """
        # Fast path: nothing to do if keep_ratio == 1 and min_tokens is large
        if self.config.keep_ratio >= 0.999:
            return embeddings, artifacts

        compressed_embeddings: list[torch.Tensor] = []

        # Try to grab optional global artifacts (per-doc lists)
        idf_docs = artifacts.get("idf", None)
        token_weight_docs = artifacts.get("token_weights", None)

        # Compute IDF if needed and not provided
        if self.config.use_idf and idf_docs is None:
            input_ids = artifacts.get("input_ids", None)
            if input_ids is not None:
                idf_docs = self._compute_document_idf(input_ids, show_progress=False)

        for doc_idx, doc_emb in enumerate(embeddings):
            device = doc_emb.device
            n_tokens, dim = doc_emb.shape

            if n_tokens == 0:
                compressed_embeddings.append(doc_emb)
                continue

            # Determine how many tokens we want to keep
            protected = min(self.config.protected_tokens, n_tokens)
            # Non-protected range: [protected, n_tokens)
            n_non_protected = max(n_tokens - protected, 0)

            if n_non_protected <= 0:
                # Nothing to prune
                compressed_embeddings.append(doc_emb)
                continue

            # Target total tokens after pruning (incl. protected)
            target_total = max(
                self.config.min_tokens,
                protected + int(self.config.keep_ratio * n_non_protected),
            )
            target_total = min(target_total, n_tokens)
            target_non_protected = max(target_total - protected, 0)

            if target_non_protected >= n_non_protected:
                # We are not actually pruning this doc
                compressed_embeddings.append(doc_emb)
                continue

            # Slice non-protected tokens
            non_protected_emb = doc_emb[protected:, :]  # [n_non_protected, dim]

            # Build per-token artifact vectors for this doc
            if idf_docs is not None:
                idf_vec_full = self._get_artifact_vector(
                    idf_docs[doc_idx],
                    length=n_tokens,
                    device=device,
                    dtype=torch.float32,
                    default_val=0.0,
                )
                idf_vec = idf_vec_full[protected:]
            else:
                idf_vec = None

            if token_weight_docs is not None:
                token_weight_full = self._get_artifact_vector(
                    token_weight_docs[doc_idx],
                    length=n_tokens,
                    device=device,
                    dtype=torch.float32,
                    default_val=0.0,
                )
                token_weight_vec = token_weight_full[protected:]
            else:
                token_weight_vec = None

            # Compute importance scores for non-protected tokens
            scores_full = self._compute_importance_scores(
                doc_embeddings=non_protected_emb,
                idf_vec=idf_vec,
                token_weight_vec=token_weight_vec,
            )  # [n_non_protected]

            # Choose top-K tokens by importance
            k = target_non_protected
            topk_scores, topk_indices = torch.topk(
                scores_full, k=k, largest=True, sorted=False
            )

            # Sort selected indices to keep original order
            topk_indices_sorted, _ = torch.sort(topk_indices)
            # Map them back to original positions
            keep_indices = torch.cat(
                [
                    torch.arange(0, protected, device=device),
                    topk_indices_sorted + protected,
                ],
                dim=0,
            )

            # Final embeddings for this doc
            doc_emb_pruned = doc_emb[keep_indices]
            compressed_embeddings.append(doc_emb_pruned)

        # Update artifacts: shape-matched artifacts are pruned per-doc,
        # metadata artifacts are passed through untouched.
        updated_artifacts: CompressionArtifacts = {}

        for name, value in artifacts.items():
            # Metadata: keep as-is
            if not isinstance(value, list):
                updated_artifacts[name] = value
                continue

            # Shape-matched: per-doc list of sequences
            pruned_value_list = []
            for doc_idx, artifact_tokens in enumerate(value):
                doc_emb = embeddings[doc_idx]
                n_tokens = doc_emb.shape[0]

                if n_tokens == 0:
                    pruned_value_list.append(artifact_tokens)
                    continue

                # Recompute the keep_indices for this doc exactly as above.
                protected = min(self.config.protected_tokens, n_tokens)
                n_non_protected = max(n_tokens - protected, 0)

                if n_non_protected <= 0:
                    # Only protected tokens exist
                    keep_indices = torch.arange(0, n_tokens)
                else:
                    target_total = max(
                        self.config.min_tokens,
                        protected + int(self.config.keep_ratio * n_non_protected),
                    )
                    target_total = min(target_total, n_tokens)
                    target_non_protected = max(target_total - protected, 0)

                    if target_non_protected >= n_non_protected:
                        keep_indices = torch.arange(0, n_tokens)
                    else:
                        device = doc_emb.device
                        non_protected_emb = doc_emb[protected:, :]

                        # Rebuild per-token artifact vectors (optional)
                        if idf_docs is not None and self.config.use_idf:
                            idf_vec_full = self._get_artifact_vector(
                                idf_docs[doc_idx],
                                length=n_tokens,
                                device=device,
                                dtype=torch.float32,
                                default_val=0.0,
                            )
                            idf_vec = idf_vec_full[protected:]
                        else:
                            idf_vec = None

                        if token_weight_docs is not None and self.config.use_token_weights:
                            token_weight_full = self._get_artifact_vector(
                                token_weight_docs[doc_idx],
                                length=n_tokens,
                                device=device,
                                dtype=torch.float32,
                                default_val=0.0,
                            )
                            token_weight_vec = token_weight_full[protected:]
                        else:
                            token_weight_vec = None

                        scores_full = self._compute_importance_scores(
                            doc_embeddings=non_protected_emb,
                            idf_vec=idf_vec,
                            token_weight_vec=token_weight_vec,
                        )

                        k = target_non_protected
                        _, topk_indices = torch.topk(
                            scores_full, k=k, largest=True, sorted=False
                        )
                        topk_indices_sorted, _ = torch.sort(topk_indices)
                        keep_indices = torch.cat(
                            [
                                torch.arange(0, protected, device=device),
                                topk_indices_sorted + protected,
                            ],
                            dim=0,
                        )

                # Now apply keep_indices to this artifact
                if isinstance(artifact_tokens, torch.Tensor):
                    # Tensor case
                    if artifact_tokens.numel() < n_tokens:
                        # Pad if needed
                        pad_len = n_tokens - artifact_tokens.numel()
                        pad = artifact_tokens[-1:].repeat(pad_len)
                        artifact_tokens = torch.cat([artifact_tokens, pad], dim=0)
                    pruned_tokens = artifact_tokens[keep_indices.cpu()]
                else:
                    # Treat as a sequence (list, etc.)
                    tokens_list = list(artifact_tokens)
                    if len(tokens_list) < n_tokens:
                        tokens_list = tokens_list + [tokens_list[-1]] * (
                            n_tokens - len(tokens_list)
                        )
                    pruned_tokens = [tokens_list[i.item()] for i in keep_indices.cpu()]

                pruned_value_list.append(pruned_tokens)

            updated_artifacts[name] = pruned_value_list

        return compressed_embeddings, updated_artifacts
