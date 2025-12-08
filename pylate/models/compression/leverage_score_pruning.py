from .base import *

@dataclass
class LeverageScorePruningConfig(CompressionStrategyConfigBase):
    """
    Configuration for leverage score-based pruning.

    Prunes tokens with low statistical leverage scores, i.e., tokens that are less
    important for approximating the embedding matrix's column space. This is a
    query-independent method based on randomized numerical linear algebra.

    Attributes
    ----------
    top_k
        Number of tokens (after the protected prefix) to **prune/remove**. Mutually exclusive with ``threshold``.
        Removes tokens with the **lowest leverage scores** (least geometrically important tokens).
        This is the number of token **occurrences** to prune per document.
        Note: This is the number to REMOVE, not the number to KEEP.
    threshold
        Prune tokens whose leverage score is < threshold. Mutually exclusive with ``top_k``.
        Lower leverage score = less important for matrix approximation.
    protected_tokens
        Number of leading tokens to always retain (CLS / prefixes).
    min_tokens
        Minimum total number of tokens per document after pruning (including protected tokens).
        Prevents over-pruning. Defaults to 1 (at least keep protected tokens).
    projection_dim
        Dimension for Johnson-Lindenstrauss random projection. Lower values are faster
        but less accurate. Defaults to 64. Must be positive and typically much smaller
        than the embedding dimension.
    center_embeddings
        If True, centers embeddings (zero-mean) before computing leverage scores.
        This can improve numerical stability. Defaults to True.
    normalize_scores
        If True, normalizes leverage scores to z-scores (mean=0, std=1) per document.
        This ensures consistent thresholding across documents. Defaults to True.
    track_pruned_tokens
        If True, tracks which tokens were pruned from each document. Access via
        ``strategy.get_pruned_tokens()`` after encoding. Defaults to False.
    show_progress_bar
        If True, shows a progress bar during pruning. Defaults to False.
    """

    top_k: Optional[int] = None
    threshold: Optional[float] = None
    keep_ratio: Optional[float] = None  # fraction of non-protected tokens to keep
    protected_tokens: int = 1
    min_tokens: int = 1
    projection_dim: int = 64
    center_embeddings: bool = True
    normalize_scores: bool = True
    track_pruned_tokens: bool = False
    show_progress_bar: bool = False

    def __post_init__(self) -> None:
        provided = [p is not None for p in (self.top_k, self.threshold, self.keep_ratio)]
        if not any(provided):
            raise ValueError("Leverage score pruning requires one of `top_k`, `threshold`, or `keep_ratio`.")
        if sum(provided) > 1:
            raise ValueError(
                "Provide only one of `top_k`, `threshold`, or `keep_ratio` for leverage score pruning."
            )
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("`top_k` must be a non-negative integer.")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("`threshold` must be a finite float.")
        if self.keep_ratio is not None and not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError("`keep_ratio` must be in (0, 1].")
        if self.projection_dim <= 0:
            raise ValueError("`projection_dim` must be a positive integer.")
        if self.min_tokens < 1:
            raise ValueError("`min_tokens` must be at least 1.")
        if self.protected_tokens < 0:
            raise ValueError("`protected_tokens` must be non-negative.")

    def serialize(self) -> dict:
        """
        Serialize this configuration to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "top_k": self.top_k,
            "threshold": self.threshold,
            "keep_ratio": self.keep_ratio,
            "protected_tokens": self.protected_tokens,
            "min_tokens": self.min_tokens,
            "projection_dim": self.projection_dim,
            "center_embeddings": self.center_embeddings,
            "normalize_scores": self.normalize_scores,
            "track_pruned_tokens": self.track_pruned_tokens,
            "show_progress_bar": self.show_progress_bar,
        }

    @property
    def strategy_type(self) -> str:
        return "leverage_score_pruning"

class LeverageScorePruningStrategy(CompressionStrategy):
    """
    Leverage score-based pruning strategy that removes tokens with low statistical leverage scores.

    This strategy uses statistical leverage scores from randomized numerical linear algebra
    to identify and prune tokens that are less important for approximating the embedding
    matrix's column space. Unlike attention-based pruning, this is query-independent and
    based purely on the geometric structure of the embeddings.

    The method uses Johnson-Lindenstrauss random projection for computational efficiency,
    reducing the embedding dimension before computing leverage scores via SVD.
    """

    required_artifacts: list[str] = []  # Leverage scores computed directly from embeddings

    def __init__(self, config: LeverageScorePruningConfig):
        """
        Initialize the leverage score pruning strategy.

        Parameters
        ----------
        config
            Leverage score pruning configuration specifying top_k/threshold, protected_tokens, etc.
        """
        self.config = config
        self._pruned_tokens: Optional[list[list[int]]] = None  # Track pruned tokens if requested

    @property
    def name(self) -> str:
        return "LeverageScorePruning"

    @property
    def strategy_type(self) -> str:
        """Strategy type identifier for serialization."""
        return "leverage-score"

    def get_pruned_tokens(self) -> Optional[list[list[int]]]:
        """
        Get the list of pruned token IDs for each document.

        Returns
        -------
        Optional[list[list[int]]]
            List of pruned token IDs per document, or None if tracking was not enabled
        """
        return self._pruned_tokens

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
    def from_dict(cls, data: dict) -> "LeverageScorePruningStrategy":
        """
        Create a strategy instance from a serialized dictionary.

        Parameters
        ----------
        data
            Dictionary containing serialized strategy data

        Returns
        -------
        LeverageScorePruningStrategy
            Deserialized strategy instance
        """
        config_data = data.get("config", {})
        config = LeverageScorePruningConfig(
            top_k=config_data.get("top_k"),
            threshold=config_data.get("threshold"),
            keep_ratio=config_data.get("keep_ratio"),
            protected_tokens=config_data.get("protected_tokens", 1),
            min_tokens=config_data.get("min_tokens", 1),
            projection_dim=config_data.get("projection_dim", 64),
            center_embeddings=config_data.get("center_embeddings", True),
            normalize_scores=config_data.get("normalize_scores", True),
            track_pruned_tokens=config_data.get("track_pruned_tokens", False),
            show_progress_bar=config_data.get("show_progress_bar", False),
        )
        return cls(config)

    def _compute_leverage_scores(
        self,
        doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute statistical leverage scores for a single document's embeddings.

        Uses Johnson-Lindenstrauss random projection followed by SVD to efficiently
        compute leverage scores. The leverage score of token i is ||U_i||^2 where U
        is the left singular vectors of the embedding matrix.

        Parameters
        ----------
        doc_embeddings
            Embedding tensor for one document, shape (num_tokens, embedding_dim)

        Returns
        -------
        torch.Tensor
            Leverage scores for each token, shape (num_tokens,)
        """
        n, d = doc_embeddings.shape  # n = num_tokens, d = embedding_dim

        # Handle edge case: if we have fewer tokens than projection_dim
        k = min(self.config.projection_dim, d, n)

        # Center embeddings if requested
        if self.config.center_embeddings:
            embeddings_centered = doc_embeddings - doc_embeddings.mean(dim=0, keepdim=True)
        else:
            embeddings_centered = doc_embeddings

        # Johnson-Lindenstrauss random projection: (n, d) @ (d, k) -> (n, k)
        # This reduces computational cost from O(n*d^2) to O(n*k^2 + k^3)
        R = torch.randn(d, k, device=doc_embeddings.device, dtype=doc_embeddings.dtype) * (1.0 / math.sqrt(k))
        embeddings_projected = torch.matmul(embeddings_centered, R)  # (n, k)

        # Compute Gram matrix in reduced space: K^T @ K where K is (n, k)
        # gram shape: (k, k)
        gram = torch.matmul(embeddings_projected.T, embeddings_projected)

        # SVD of Gram matrix: gram = V @ diag(S) @ V^T
        # We use float32 for numerical stability
        try:
            V, S, _ = torch.linalg.svd(gram.to(torch.float32), full_matrices=False)
        except RuntimeError:
            # If SVD fails, fall back to eigendecomposition
            S, V = torch.linalg.eigh(gram.to(torch.float32))
            # Sort in descending order (eigh returns ascending)
            S = S.flip(dims=[-1])
            V = V.flip(dims=[-1])

        # Filter out near-zero singular values for numerical stability
        # Use relative threshold based on largest singular value
        max_singular_value = S.max()
        valid_mask = S >= max_singular_value * 1e-6

        if valid_mask.sum() == 0:
            # All singular values are too small, return uniform scores
            return torch.ones(n, device=doc_embeddings.device, dtype=doc_embeddings.dtype)

        # Compute pseudo-inverse: S^{-1/2}
        S_filtered = S[valid_mask]
        V_filtered = V[:, valid_mask]
        S_inv_sqrt = S_filtered.rsqrt()  # 1 / sqrt(S)

        # Compute U = K @ V @ S^{-1/2}
        # U shape: (n, num_valid_singular_values)
        U = torch.matmul(
            embeddings_projected,
            (V_filtered * S_inv_sqrt.unsqueeze(0)).to(doc_embeddings.dtype)
        )

        # Leverage scores: ||U_i||^2 for each row i
        leverage_scores = (U * U).sum(dim=-1)  # (n,)

        return leverage_scores

    def _get_positions_to_prune(
        self,
        leverage_scores: torch.Tensor,
        num_tokens: int,
    ) -> set[int]:
        """
        Determine which token positions to prune based on leverage scores.

        Parameters
        ----------
        leverage_scores
            Leverage scores for each token position, shape (num_tokens,)
        num_tokens
            Total number of tokens in the document

        Returns
        -------
        set[int]
            Set of token positions (indices) to prune in this document
        """
        # Normalize scores if requested
        if self.config.normalize_scores:
            # Z-score normalization: (x - mean) / std
            mean = leverage_scores.mean()
            std = leverage_scores.std()
            if std > 1e-8:  # Avoid division by zero
                leverage_scores = (leverage_scores - mean) / std

        # Score each token occurrence (after protected tokens)
        token_scores: list[tuple[int, float]] = []  # (position, score)

        for pos in range(self.config.protected_tokens, len(leverage_scores)):
            score = leverage_scores[pos].item()
            token_scores.append((pos, score))

        # Determine max number of tokens we can prune while respecting min_tokens
        max_prunable = max(0, num_tokens - self.config.min_tokens)

        # Select tokens to prune based on top_k/keep_ratio or threshold
        if self.config.top_k is not None or self.config.keep_ratio is not None:
            # Sort by score (ascending) and take top_k lowest
            sorted_tokens = sorted(token_scores, key=lambda x: x[1])
            # Respect both top_k and min_tokens constraint
            if self.config.keep_ratio is not None:
                keep = int(math.ceil(self.config.keep_ratio * len(sorted_tokens)))
                num_to_prune = max(0, len(sorted_tokens) - keep)
            else:
                num_to_prune = self.config.top_k
            num_to_prune = min(num_to_prune, len(sorted_tokens), max_prunable)
            positions_to_prune = {pos for pos, _ in sorted_tokens[:num_to_prune]}
        else:
            # Prune tokens with score < threshold, but respect min_tokens
            candidates = [
                (pos, score) for pos, score in token_scores
                if score < self.config.threshold
            ]
            # Sort by score (ascending) to prune lowest scores first
            candidates_sorted = sorted(candidates, key=lambda x: x[1])
            num_to_prune = min(len(candidates_sorted), max_prunable)
            positions_to_prune = {pos for pos, _ in candidates_sorted[:num_to_prune]}

        return positions_to_prune

    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply leverage score-based pruning to embeddings and update artifacts.

        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Can optionally contain "input_ids" for tracking pruned tokens.

        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pruned_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with pruned embeddings.
            Metadata artifacts are passed through unchanged.
        """
        # Get input_ids if available (for tracking pruned tokens)
        input_ids = artifacts.get("input_ids")
        if input_ids is not None and not isinstance(input_ids, list):
            raise ValueError("input_ids artifact must be a list of tensors if provided")
        if input_ids is not None and len(input_ids) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(input_ids)} input_ids but {len(embeddings)} embeddings"
            )

        # Initialize pruned tokens tracking if requested
        # Check if we're in parallel mode (pruned_tokens already pre-allocated)
        start_idx = artifacts.get("_batch_start_idx", 0)
        is_parallel_mode = (
            self.config.track_pruned_tokens and
            self._pruned_tokens is not None and
            isinstance(self._pruned_tokens, list) and
            len(self._pruned_tokens) > len(embeddings)  # Pre-allocated for parallel
        )

        if self.config.track_pruned_tokens and not is_parallel_mode:
            # Sequential mode: start fresh
            self._pruned_tokens = []
        elif not self.config.track_pruned_tokens:
            self._pruned_tokens = None

        # Prune tokens based on leverage scores
        pruned_embeddings: list[torch.Tensor] = []
        updated_input_ids: list[torch.Tensor] = []
        keep_masks_per_doc: list[torch.Tensor] = []  # Cache masks to avoid recomputing for artifacts

        criterion_str = (
            f"top_k={self.config.top_k}"
            if self.config.top_k is not None
            else f"threshold={self.config.threshold}"
            if self.config.threshold is not None
            else f"keep_ratio={self.config.keep_ratio}"
        )
        iterator = tqdm(
            enumerate(embeddings),
            desc=f"Leverage score pruning ({criterion_str})",
            total=len(embeddings),
            disable=not self.config.show_progress_bar,
        )

        for batch_doc_idx, doc_embeddings in iterator:
            num_tokens = doc_embeddings.shape[0]

            # Get input_ids for this document if available
            doc_input_ids = input_ids[batch_doc_idx] if input_ids is not None else None

            # Compute leverage scores for this document ONCE
            leverage_scores = self._compute_leverage_scores(doc_embeddings)

            # Get positions to prune (respects min_tokens constraint)
            positions_to_prune = self._get_positions_to_prune(leverage_scores, num_tokens)

            # Create mask: keep tokens not in positions_to_prune
            keep_mask = []
            pruned_token_ids = []

            for pos in range(num_tokens):
                if pos < self.config.protected_tokens:
                    # Always keep protected tokens
                    keep_mask.append(True)
                elif pos in positions_to_prune:
                    # Prune this token
                    keep_mask.append(False)
                    if self.config.track_pruned_tokens and doc_input_ids is not None:
                        pruned_token_ids.append(doc_input_ids[pos].item())
                else:
                    # Keep this token
                    keep_mask.append(True)

            # Convert to tensor and cache on CPU to avoid device issues
            keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool)
            keep_masks_per_doc.append(keep_mask_tensor)

            # Apply mask to embeddings
            keep_mask_on_device = keep_mask_tensor.to(doc_embeddings.device)
            pruned_doc_embeddings = doc_embeddings[keep_mask_on_device]
            pruned_embeddings.append(pruned_doc_embeddings)

            # Update input_ids if available
            if doc_input_ids is not None:
                pruned_doc_tokens = doc_input_ids[keep_mask_on_device]
                updated_input_ids.append(pruned_doc_tokens)

            if self.config.track_pruned_tokens:
                if is_parallel_mode:
                    # Parallel mode: set at correct index
                    self._pruned_tokens[start_idx + batch_doc_idx] = pruned_token_ids
                else:
                    # Sequential mode: append
                    self._pruned_tokens.append(pruned_token_ids)

        # Update artifacts using cached masks (no recomputation!)
        updated_artifacts: CompressionArtifacts = {}

        for artifact_name, artifact_value in artifacts.items():
            if artifact_name == "input_ids" and input_ids is not None:
                # Already updated above
                updated_artifacts[artifact_name] = updated_input_ids
            elif artifact_name.startswith("_"):
                # Internal metadata (like _batch_start_idx): skip
                continue
            elif not isinstance(artifact_value, list):
                # Metadata artifact: pass through unchanged
                updated_artifacts[artifact_name] = artifact_value
            else:
                # Shape-matched artifact: apply cached masks
                pruned_artifacts = []
                for doc_idx, artifact_tokens in enumerate(artifact_value):
                    mask_cpu = keep_masks_per_doc[doc_idx]

                    if isinstance(artifact_tokens, torch.Tensor):
                        # Tensor artifact: use boolean indexing
                        mask = mask_cpu.to(artifact_tokens.device)
                        # Truncate mask if artifact is shorter than expected
                        if artifact_tokens.shape[0] < mask.shape[0]:
                            mask = mask[: artifact_tokens.shape[0]]
                        pruned_tokens = artifact_tokens[mask]
                    else:
                        # List/sequence artifact: filter by mask
                        mask_list = mask_cpu.tolist()
                        # Truncate mask if artifact is shorter
                        mask_list = mask_list[: len(artifact_tokens)]
                        pruned_tokens = [
                            t for t, m in zip(artifact_tokens, mask_list) if m
                        ]

                    pruned_artifacts.append(pruned_tokens)

                updated_artifacts[artifact_name] = pruned_artifacts

        return pruned_embeddings, updated_artifacts

    def compress_parallel(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
        batch_size: int = 100,
        num_workers: Optional[int] = None,
        show_progress: bool = False,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply leverage score-based pruning in parallel.

        This method parallelizes the per-document pruning operations using the base
        class parallel implementation.

        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts.
        batch_size
            Number of documents to process in each batch. Defaults to 100.
        num_workers
            Number of worker threads to use. If None, defaults to min(batch_size, number of documents, 8).
        show_progress
            If True, shows a progress bar during parallel compression. Defaults to False.

        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pruned_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with pruned embeddings.
            Metadata artifacts are passed through unchanged.
        """
        # Initialize pruned tokens tracking if requested
        if self.config.track_pruned_tokens:
            self._pruned_tokens = [None] * len(embeddings)
        else:
            self._pruned_tokens = None

        # Call the base class compress_parallel, which will call compress() on each batch
        result_emb, result_art = super().compress_parallel(embeddings, artifacts, batch_size, num_workers, show_progress)

        # If tracking pruned tokens, ensure all were collected
        if self.config.track_pruned_tokens and self._pruned_tokens is not None:
            if None in self._pruned_tokens:
                raise RuntimeError("Some pruned tokens were not collected during parallel compression")

        return result_emb, result_art
