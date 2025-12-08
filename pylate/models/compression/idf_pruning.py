from .base import *

@dataclass
class IDFPruningConfig(CompressionStrategyConfigBase):
    """
    Configuration for IDF-based pruning.

    Prunes tokens with low IDF scores, i.e., tokens that occur frequently across documents
    (high document frequency) and are therefore less informative.

    Attributes
    ----------
    mode
        ``"global"`` identifies the k unique token types with the lowest IDF scores and prunes
        **all occurrences** of those k token types across all documents in the batch. Different
        documents may have different numbers of tokens removed depending on how many times those
        token types appear.
        ``"document"`` prunes k lowest-IDF token occurrences from each document independently.
    top_k
        Number of tokens (after the protected prefix) to prune/remove. Mutually exclusive with ``threshold``.
        Removes tokens with the **lowest IDF scores** (highest document frequency, less informative).
        In "global" mode, this is the number of unique token **types** to identify and prune all
        occurrences of across the entire batch.
        In "document" mode, this is the number of token **occurrences** to prune per document.
    threshold
        Prune tokens whose IDF score is < threshold. Mutually exclusive with ``top_k``.
        Lower IDF = more common across documents = less informative.
    protected_tokens
        Number of leading tokens to always retain (CLS / prefixes).
    use_tfidf
        If True, uses TF-IDF scoring (considers in-document frequency).
        If False, uses only IDF scoring. Defaults to False.

    track_pruned_tokens
        If True, tracks which tokens were pruned from each document. Access via
        ``strategy.get_pruned_tokens()`` after encoding. Defaults to False.
    show_progress_bar
        If True, shows a progress bar during pruning. Defaults to False.
    """

    mode: Literal["global", "document"] = "document"
    top_k: Optional[int] = None
    threshold: Optional[float] = None
    keep_ratio: Optional[float] = None
    protected_tokens: int = 1
    use_tfidf: bool = False
    track_pruned_tokens: bool = False
    ignore_token_ids: Optional[Collection[int]] = None
    show_progress_bar: bool = False

    def __post_init__(self) -> None:
        provided = [p is not None for p in (self.top_k, self.threshold, self.keep_ratio)]
        if not any(provided):
            raise ValueError("IDF pruning requires one of `top_k`, `threshold`, or `keep_ratio`.")
        if sum(provided) > 1:
            raise ValueError(
                "Provide only one of `top_k`, `threshold`, or `keep_ratio` for IDF pruning."
            )
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("`top_k` must be a positive integer.")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("`threshold` must be a finite float.")
        if self.keep_ratio is not None and not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError("`keep_ratio` must be in (0, 1].")

    def serialize(self) -> dict:
        """
        Serialize this configuration to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        result = {
            "mode": self.mode,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "keep_ratio": self.keep_ratio,
            "protected_tokens": self.protected_tokens,
            "use_tfidf": self.use_tfidf,
            "track_pruned_tokens": self.track_pruned_tokens,
            "show_progress_bar": self.show_progress_bar,
        }
        # Convert set to list for JSON serialization
        if self.ignore_token_ids is not None:
            result["ignore_token_ids"] = list(self.ignore_token_ids)
        return result
    
    @property
    def strategy_type(self) -> str:
        return "idf_pruning"


class IDFPruningStrategy(CompressionStrategy):
    """
    IDF-based pruning strategy that removes tokens with low IDF scores.
    
    This strategy computes TF-IDF statistics from input_ids and prunes tokens
    that occur frequently across documents (high document frequency, low IDF),
    which are less informative for retrieval.
    
    Supports two modes:
    - "global": Identifies low-scoring token types globally and prunes all occurrences
    - "document": Prunes low-scoring tokens independently per document
    """
    
    required_artifacts: list[str] = ["input_ids"]  # Requires input_ids to compute TF-IDF stats
    
    def __init__(self, config: IDFPruningConfig):
        """
        Initialize the IDF pruning strategy.
        
        Parameters
        ----------
        config
            IDF pruning configuration specifying mode, top_k/threshold, protected_tokens, etc.
        """
        self.config = config
        self._pruned_tokens: Optional[list[list[int]]] = None  # Track pruned tokens if requested
    
    @property
    def name(self) -> str:
        """Name of this compression strategy."""
        mode_str = self.config.mode
        criterion_str = (
            f"topk-{self.config.top_k}"
            if self.config.top_k
            else f"threshold-{self.config.threshold}"
            if self.config.threshold is not None
            else f"keep_ratio-{self.config.keep_ratio}"
        )
        return f"idf_pruning_mode-{mode_str}_{criterion_str}"
    
    @property
    def strategy_type(self) -> str:
        """Strategy type identifier for serialization."""
        return "idf_pruning"
    
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
    def from_dict(cls, data: dict) -> "IDFPruningStrategy":
        """
        Create a strategy instance from a serialized dictionary.
        
        Parameters
        ----------
        data
            Dictionary containing serialized strategy data
            
        Returns
        -------
        IDFPruningStrategy
            Deserialized strategy instance
        """
        config_data = data.get("config", {})
        config = IDFPruningConfig(
            mode=config_data.get("mode", "document"),
            top_k=config_data.get("top_k"),
            threshold=config_data.get("threshold"),
            keep_ratio=config_data.get("keep_ratio"),
            protected_tokens=config_data.get("protected_tokens", 1),
            use_tfidf=config_data.get("use_tfidf", False),
            track_pruned_tokens=config_data.get("track_pruned_tokens", False),
            ignore_token_ids=set(config_data.get("ignore_token_ids", [])) if config_data.get("ignore_token_ids") else None,
            show_progress_bar=config_data.get("show_progress_bar", False),
        )
        return cls(config)
    
    def get_pruned_tokens(self) -> Optional[list[list[int]]]:
        """
        Get the list of pruned tokens for each document.
        
        Returns
        -------
        Optional[list[list[int]]]
            List of pruned token IDs per document, or None if tracking is disabled.
            Only available if track_pruned_tokens=True was set in config.
        """
        return self._pruned_tokens
    
    def _compute_tfidf_stats(self, input_ids: list[torch.Tensor]) -> TokenTFIDFStats:
        """
        Compute TF-IDF statistics from input_ids.
        
        Parameters
        ----------
        input_ids
            List of input_id tensors (one per document)
        
        Returns
        -------
        TokenTFIDFStats
            Fitted TF-IDF statistics
        """
        # Convert tensors to lists for TokenTFIDFStats
        tokenized_docs = [doc_input_ids.cpu().tolist() for doc_input_ids in input_ids]
        
        # Compute TF-IDF statistics
        stats = TokenTFIDFStats(num_docs=len(tokenized_docs))
        stats.fit(tokenized_docs, show_progress=self.config.show_progress_bar)
        return stats
    
    def _get_token_score(self, stats: TokenTFIDFStats, doc_idx: int, token_id: int) -> float:
        """
        Get the score for a token (IDF or TF-IDF).
        
        Parameters
        ----------
        stats
            TF-IDF statistics
        doc_idx
            Document index
        token_id
            Token ID
        
        Returns
        -------
        float
            Token score (IDF or TF-IDF)
        """
        if self.config.use_tfidf:
            return stats.get_tfidf(doc_idx, token_id)
        else:
            return stats.get_idf(token_id)
    
    def _get_tokens_to_prune_global(
        self,
        stats: TokenTFIDFStats,
        input_ids: Optional[list[torch.Tensor]] = None,
    ) -> set[int]:
        """
        Identify token types to prune globally based on lowest scores.
        
        Parameters
        ----------
        stats
            TF-IDF statistics (computed on all documents)
        input_ids
            Optional list of input_id tensors. If provided, used to check protected tokens.
            If None, computes from stats alone (assumes no protected token filtering needed).
        
        Returns
        -------
        set[int]
            Set of token IDs to prune globally
        """
        # Collect all unique token types and their scores
        # Iterate over all tokens in the stats (from idf_scores which contains all tokens in corpus)
        token_scores: dict[int, float] = {}
        
        # Get all token IDs from stats
        all_token_ids = set(stats.idf_scores.keys())
        
        # If input_ids provided, we need to check protected tokens per document
        # Otherwise, just use IDF scores directly
        if input_ids is not None:
            # Check which tokens appear in protected positions
            tokens_in_protected_positions: set[int] = set()
            for doc_input_ids in input_ids:
                doc_tokens = doc_input_ids.cpu().tolist()
                protected_tokens = set(doc_tokens[:self.config.protected_tokens])
                tokens_in_protected_positions.update(protected_tokens)
            
            for token_id in all_token_ids:
                # Skip ignored tokens
                if self.config.ignore_token_ids and token_id in self.config.ignore_token_ids:
                    continue
                
                # Skip tokens that appear in protected positions (conservative approach)
                if token_id in tokens_in_protected_positions:
                    continue
                
                # For global mode with IDF, just use IDF score (doesn't depend on doc_idx)
                # For TF-IDF, we'd need to check all documents, but for now use IDF
                if self.config.use_tfidf:
                    # For TF-IDF, get minimum score across all documents where token appears
                    min_score = float('inf')
                    for doc_idx in range(stats.num_docs):
                        if token_id in stats.doc_token_counts[doc_idx]:
                            score = stats.get_tfidf(doc_idx, token_id)
                            min_score = min(min_score, score)
                    if min_score == float('inf'):
                        continue  # Token doesn't appear in any document
                    token_scores[token_id] = min_score
                else:
                    # Just use IDF score (same for all documents)
                    token_scores[token_id] = stats.get_idf(token_id)
        else:
            # No input_ids provided - compute from stats alone
            # Skip ignored tokens, but can't check protected tokens without input_ids
            for token_id in all_token_ids:
                if self.config.ignore_token_ids and token_id in self.config.ignore_token_ids:
                    continue
                
                # Use IDF score (or minimum TF-IDF if use_tfidf is True)
                if self.config.use_tfidf:
                    # For TF-IDF, get minimum score across all documents
                    min_score = float('inf')
                    for doc_idx in range(stats.num_docs):
                        if token_id in stats.doc_token_counts[doc_idx]:
                            score = stats.get_tfidf(doc_idx, token_id)
                            min_score = min(min_score, score)
                    if min_score == float('inf'):
                        continue
                    token_scores[token_id] = min_score
                else:
                    token_scores[token_id] = stats.get_idf(token_id)
        
        # Select tokens to prune based on top_k/keep_ratio or threshold
        if self.config.top_k is not None or self.config.keep_ratio is not None:
            # Sort by score (ascending) and take top_k lowest
            sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1])
            if self.config.keep_ratio is not None:
                keep = int(math.ceil(self.config.keep_ratio * len(sorted_tokens)))
                num_to_prune = max(0, len(sorted_tokens) - keep)
            else:
                num_to_prune = self.config.top_k
            num_to_prune = min(num_to_prune, len(sorted_tokens))
            tokens_to_prune = {token_id for token_id, _ in sorted_tokens[:num_to_prune]}
        else:
            # Prune tokens with score < threshold
            tokens_to_prune = {
                token_id for token_id, score in token_scores.items()
                if score < self.config.threshold
            }
        
        return tokens_to_prune
    
    def _get_tokens_to_prune_document(
        self,
        stats: TokenTFIDFStats,
        doc_idx: int,
        doc_input_ids: torch.Tensor,
    ) -> set[int]:
        """
        Identify token occurrences to prune for a single document.
        
        Parameters
        ----------
        stats
            TF-IDF statistics
        doc_idx
            Document index
        doc_input_ids
            Input IDs tensor for this document
        
        Returns
        -------
        set[int]
            Set of token positions (indices) to prune in this document
        """
        doc_tokens = doc_input_ids.cpu().tolist()
        
        # Score each token occurrence (after protected tokens)
        token_scores: list[tuple[int, float]] = []  # (position, score)
        
        for pos in range(self.config.protected_tokens, len(doc_tokens)):
            token_id = doc_tokens[pos]
            
            # Skip ignored tokens
            if self.config.ignore_token_ids and token_id in self.config.ignore_token_ids:
                continue
            
            score = self._get_token_score(stats, doc_idx, token_id)
            token_scores.append((pos, score))
        
        # Select tokens to prune based on top_k/keep_ratio or threshold
        if self.config.top_k is not None or self.config.keep_ratio is not None:
            # Sort by score (ascending) and take top_k lowest
            # Note: ignored tokens are already excluded from token_scores, so we'll prune exactly top_k non-ignored tokens
            sorted_tokens = sorted(token_scores, key=lambda x: x[1])
            # Take min(top_k, len(sorted_tokens)) to handle cases where we have fewer candidates than top_k
            if self.config.keep_ratio is not None:
                keep = int(math.ceil(self.config.keep_ratio * len(sorted_tokens)))
                num_to_prune = max(0, len(sorted_tokens) - keep)
            else:
                num_to_prune = self.config.top_k
            num_to_prune = min(num_to_prune, len(sorted_tokens))
            positions_to_prune = {pos for pos, _ in sorted_tokens[:num_to_prune]}
        else:
            # Prune tokens with score < threshold
            positions_to_prune = {
                pos for pos, score in token_scores
                if score < self.config.threshold
            }
        
        return positions_to_prune
    
    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply IDF-based pruning to embeddings and update artifacts.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain "input_ids" as a list of tensors.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pruned_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with pruned embeddings.
            Metadata artifacts are passed through unchanged.
        
        Raises
        ------
        ValueError
            If input_ids artifact is missing
        """
        # Get input_ids from artifacts
        if "input_ids" not in artifacts:
            raise ValueError(
                "IDFPruningStrategy requires 'input_ids' artifact. "
                "Ensure input_ids are provided when encoding."
            )
        
        input_ids = artifacts["input_ids"]
        if not isinstance(input_ids, list):
            raise ValueError("input_ids artifact must be a list of tensors")
        
        if len(input_ids) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(input_ids)} input_ids but {len(embeddings)} embeddings"
            )
        
        # Validate all input_ids are tensors
        for idx, doc_input_ids in enumerate(input_ids):
            if not isinstance(doc_input_ids, torch.Tensor):
                raise ValueError(f"input_ids[{idx}] must be a torch.Tensor, got {type(doc_input_ids)}")
        
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
        
        # Compute or use cached TF-IDF statistics from artifacts
        if "tfidf_stats" in artifacts:
            stats = artifacts["tfidf_stats"]
        else:
            stats = self._compute_tfidf_stats(input_ids)
        
        # Determine tokens to prune based on mode
        # Store keep masks for artifact updates
        keep_masks: list[list[bool]] = []
        
        if self.config.mode == "global":
            # Global mode: identify token types to prune globally
            # Check if pre-computed tokens are available (from parallel processing)
            if "_tokens_to_prune_global" in artifacts:
                tokens_to_prune_global = artifacts["_tokens_to_prune_global"]
            else:
                # Sequential mode: compute from current input_ids
                tokens_to_prune_global = self._get_tokens_to_prune_global(stats, input_ids)
            
            # Prune all occurrences of these token types from all documents
            pruned_embeddings = []
            updated_input_ids = []
            
            criterion_str = (
                f"top_k={self.config.top_k}"
                if self.config.top_k
                else f"threshold={self.config.threshold}"
                if self.config.threshold is not None
                else f"keep_ratio={self.config.keep_ratio}"
            )
            doc_pairs = list(zip(embeddings, input_ids))
            iterator = tqdm(
                enumerate(doc_pairs),
                desc=f"IDF pruning (global, {criterion_str})",
                total=len(embeddings),
                disable=not self.config.show_progress_bar,
            )
            
            for batch_doc_idx, (doc_embeddings, doc_input_ids) in iterator:
                doc_tokens = doc_input_ids.cpu().tolist()
                device = doc_input_ids.device
                dtype = doc_input_ids.dtype
                
                # Create mask: keep tokens that are not in tokens_to_prune_global
                # Always keep protected tokens
                keep_mask = []
                pruned_token_ids = []
                
                for pos, token_id in enumerate(doc_tokens):
                    if pos < self.config.protected_tokens:
                        # Always keep protected tokens
                        keep_mask.append(True)
                    elif token_id in tokens_to_prune_global:
                        # Prune this token
                        keep_mask.append(False)
                        if self.config.track_pruned_tokens:
                            pruned_token_ids.append(token_id)
                    else:
                        # Keep this token
                        keep_mask.append(True)
                
                # Apply mask to embeddings and input_ids
                # Ensure mask length matches embedding length
                if len(keep_mask) != doc_embeddings.shape[0]:
                    raise ValueError(
                        f"Mask length ({len(keep_mask)}) does not match embedding length "
                        f"({doc_embeddings.shape[0]}) for document {start_idx + batch_doc_idx}"
                    )
                keep_mask_tensor = torch.tensor(keep_mask, device=doc_embeddings.device, dtype=torch.bool)
                pruned_doc_embeddings = doc_embeddings[keep_mask_tensor]
                pruned_doc_tokens = torch.tensor(
                    [token for token, keep in zip(doc_tokens, keep_mask) if keep],
                    device=device,
                    dtype=dtype
                )
                
                pruned_embeddings.append(pruned_doc_embeddings)
                updated_input_ids.append(pruned_doc_tokens)
                
                if self.config.track_pruned_tokens:
                    if is_parallel_mode:
                        # Parallel mode: set at correct index
                        self._pruned_tokens[start_idx + batch_doc_idx] = pruned_token_ids
                    else:
                        # Sequential mode: append
                        self._pruned_tokens.append(pruned_token_ids)
                
                # Store keep mask for artifact updates
                keep_masks.append(keep_mask)
        
        else:  # document mode
            # Document mode: prune independently per document
            pruned_embeddings = []
            updated_input_ids = []
            
            criterion_str = (
                f"top_k={self.config.top_k}"
                if self.config.top_k
                else f"threshold={self.config.threshold}"
                if self.config.threshold is not None
                else f"keep_ratio={self.config.keep_ratio}"
            )
            iterator = tqdm(
                zip(embeddings, input_ids),
                desc=f"IDF pruning (document, {criterion_str})",
                total=len(embeddings),
                disable=not self.config.show_progress_bar,
            )
            
            for batch_doc_idx, (doc_embeddings, doc_input_ids) in enumerate(iterator):
                # Use global document index for stats lookup
                # Get start index from artifacts if available (for parallel processing)
                start_idx = artifacts.get("_batch_start_idx", 0)
                global_doc_idx = start_idx + batch_doc_idx
                # Get positions to prune for this document
                positions_to_prune = self._get_tokens_to_prune_document(
                    stats, global_doc_idx, doc_input_ids
                )
                
                doc_tokens = doc_input_ids.cpu().tolist()
                device = doc_input_ids.device
                dtype = doc_input_ids.dtype
                
                # Create mask: keep tokens not in positions_to_prune
                keep_mask = []
                pruned_token_ids = []
                
                for pos, token_id in enumerate(doc_tokens):
                    if pos < self.config.protected_tokens:
                        # Always keep protected tokens
                        keep_mask.append(True)
                    elif pos in positions_to_prune:
                        # Prune this token
                        keep_mask.append(False)
                        if self.config.track_pruned_tokens:
                            pruned_token_ids.append(token_id)
                    else:
                        # Keep this token
                        keep_mask.append(True)
                
                # Apply mask to embeddings and input_ids
                # Ensure mask length matches embedding length
                if len(keep_mask) != doc_embeddings.shape[0]:
                    raise ValueError(
                        f"Mask length ({len(keep_mask)}) does not match embedding length "
                        f"({doc_embeddings.shape[0]}) for document {doc_idx}"
                    )
                keep_mask_tensor = torch.tensor(keep_mask, device=doc_embeddings.device, dtype=torch.bool)
                pruned_doc_embeddings = doc_embeddings[keep_mask_tensor]
                pruned_doc_tokens = torch.tensor(
                    [token for token, keep in zip(doc_tokens, keep_mask) if keep],
                    device=device,
                    dtype=dtype
                )
                
                pruned_embeddings.append(pruned_doc_embeddings)
                updated_input_ids.append(pruned_doc_tokens)
                
                if self.config.track_pruned_tokens:
                    if is_parallel_mode:
                        # Parallel mode: set at correct index
                        self._pruned_tokens[start_idx + batch_doc_idx] = pruned_token_ids
                    else:
                        # Sequential mode: append
                        self._pruned_tokens.append(pruned_token_ids)
                
                # Store keep mask for artifact updates
                keep_masks.append(keep_mask)
        
        # Update artifacts
        updated_artifacts = {}
        for artifact_name, artifact_value in artifacts.items():
            if artifact_name == "input_ids":
                # Update input_ids to match pruned embeddings
                updated_artifacts[artifact_name] = updated_input_ids
            elif isinstance(artifact_value, list):
                # Shape-matched artifact: apply same pruning mask
                pruned_artifacts = []
                for doc_idx, artifact_tokens in enumerate(artifact_value):
                    # Use the pre-computed keep mask
                    keep_mask = keep_masks[doc_idx]
                    
                    # Apply mask to artifact (assume tensor)
                    keep_mask_tensor = torch.tensor(keep_mask, device=artifact_tokens.device, dtype=torch.bool)
                    pruned_artifact = artifact_tokens[keep_mask_tensor]
                    
                    pruned_artifacts.append(pruned_artifact)
                
                updated_artifacts[artifact_name] = pruned_artifacts
            else:
                # Metadata artifact: pass through unchanged (including tfidf_stats)
                updated_artifacts[artifact_name] = artifact_value
        
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
        Apply IDF-based pruning in parallel, computing global statistics first.
        
        This method computes TF-IDF statistics across all documents first (to ensure
        consistent results with sequential processing), then parallelizes the per-document
        pruning operations using the base class parallel implementation.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain "input_ids" as a list of tensors.
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
        # Get input_ids from artifacts to compute global stats
        if "input_ids" not in artifacts:
            raise ValueError(
                "IDFPruningStrategy requires 'input_ids' artifact. "
                "Ensure input_ids are provided when encoding."
            )
        
        input_ids = artifacts["input_ids"]
        if not isinstance(input_ids, list):
            raise ValueError("input_ids artifact must be a list of tensors")
        
        if len(input_ids) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(input_ids)} input_ids but {len(embeddings)} embeddings"
            )
        
        # Compute TF-IDF statistics from ALL input_ids first (global computation)
        # This ensures results match sequential processing
        # Add stats to artifacts so compress() can use them
        # Make a copy of artifacts to avoid modifying the input
        if "tfidf_stats" in artifacts:
            stats = artifacts["tfidf_stats"]
        else:
            stats = self._compute_tfidf_stats(input_ids)

        artifacts_with_stats = artifacts.copy()
        artifacts_with_stats["tfidf_stats"] = stats
        
        # For global mode, pre-compute tokens_to_prune_global across ALL documents
        # This must be done before batching, otherwise each batch computes different tokens
        if self.config.mode == "global":
            tokens_to_prune_global = self._get_tokens_to_prune_global(stats, input_ids)
            artifacts_with_stats["_tokens_to_prune_global"] = tokens_to_prune_global
        
        # Initialize pruned tokens tracking if requested
        if self.config.track_pruned_tokens:
            self._pruned_tokens = [None] * len(embeddings)
        else:
            self._pruned_tokens = None
        
        # Call the base class compress_parallel, which will call compress() on each batch
        # compress() will use the stats and pre-computed tokens from artifacts
        result_emb, result_art = super().compress_parallel(embeddings, artifacts_with_stats, batch_size, num_workers, show_progress)
        
        # If tracking pruned tokens, ensure all were collected
        if self.config.track_pruned_tokens and self._pruned_tokens is not None:
            if None in self._pruned_tokens:
                raise RuntimeError("Some pruned tokens were not collected during parallel compression")
        
        return result_emb, result_art
