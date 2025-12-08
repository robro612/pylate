from .base import *

@dataclass
class AttentionPruningConfig(CompressionStrategyConfigBase):
    """
    Configuration for attention-based pruning.

    Prunes tokens with low attention scores, i.e., tokens that receive less attention
    from the model and are therefore less important for the representation.

    Attributes
    ----------
    top_k
        Number of tokens (after the protected prefix) to prune/remove. Mutually exclusive with ``threshold``.
        Removes tokens with the **lowest attention scores** (least attended-to tokens).
        This is the number of token **occurrences** to prune per document.
    threshold
        Prune tokens whose attention score is < threshold. Mutually exclusive with ``top_k``.
        Lower attention = less important for the representation.
    protected_tokens
        Number of leading tokens to always retain (CLS / prefixes).
    track_pruned_tokens
        If True, tracks which tokens were pruned from each document. Access via
        ``strategy.get_pruned_tokens()`` after encoding. Defaults to False.
    show_progress_bar
        If True, shows a progress bar during pruning. Defaults to False.
    normalize_scores
        If True, normalizes attention scores with softmax after masking. This ensures
        scores sum to 1.0 per document. Defaults to False (use raw aggregated scores).
    """

    top_k: Optional[int] = None
    threshold: Optional[float] = None
    keep_ratio: Optional[float] = None
    protected_tokens: int = 1
    track_pruned_tokens: bool = False
    show_progress_bar: bool = False
    normalize_scores: bool = False

    def __post_init__(self) -> None:
        provided = [p is not None for p in (self.top_k, self.threshold, self.keep_ratio)]
        if not any(provided):
            raise ValueError("Attention pruning requires one of `top_k`, `threshold`, or `keep_ratio`.")
        if sum(provided) > 1:
            raise ValueError(
                "Provide only one of `top_k`, `threshold`, or `keep_ratio` for attention pruning."
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
        return {
            "top_k": self.top_k,
            "threshold": self.threshold,
            "keep_ratio": self.keep_ratio,
            "protected_tokens": self.protected_tokens,
            "track_pruned_tokens": self.track_pruned_tokens,
            "show_progress_bar": self.show_progress_bar,
            "normalize_scores": self.normalize_scores,
        }
    
    @property
    def strategy_type(self) -> str:
        return "attention_pruning"

class AttentionPruningStrategy(CompressionStrategy):
    """
    Attention-based pruning strategy that removes tokens with low attention scores.
    
    This strategy uses attention scores from the model's last layer to identify
    and prune tokens that receive less attention, which are less important for
    the document representation.
    """
    
    required_artifacts: list[str] = ["attention_scores"]  # Requires attention_scores to prune
    
    def __init__(self, config: AttentionPruningConfig, debug: bool = False):
        """
        Initialize the attention pruning strategy.

        Parameters
        ----------
        config
            Attention pruning configuration specifying top_k/threshold, protected_tokens, etc.
        debug
            If True, print debug information during compression
        """
        self.config = config
        self.debug = debug
        self._pruned_tokens: Optional[list[list[int]]] = None  # Track pruned tokens if requested
    
    @property
    def name(self) -> str:
        """Name of this compression strategy."""
        criterion_str = (
            f"topk-{self.config.top_k}"
            if self.config.top_k
            else f"threshold-{self.config.threshold}"
            if self.config.threshold is not None
            else f"keep_ratio-{self.config.keep_ratio}"
        )
        return f"attention_pruning_{criterion_str}"
    
    @property
    def strategy_type(self) -> str:
        """Strategy type identifier for serialization."""
        return "attention_pruning"
    
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
    def from_dict(cls, data: dict) -> "AttentionPruningStrategy":
        """
        Create a strategy instance from a serialized dictionary.
        
        Parameters
        ----------
        data
            Dictionary containing serialized strategy data
            
        Returns
        -------
        AttentionPruningStrategy
            Deserialized strategy instance
        """
        config_data = data.get("config", {})
        config = AttentionPruningConfig(
            top_k=config_data.get("top_k"),
            threshold=config_data.get("threshold"),
            keep_ratio=config_data.get("keep_ratio"),
            protected_tokens=config_data.get("protected_tokens", 1),
            track_pruned_tokens=config_data.get("track_pruned_tokens", False),
            show_progress_bar=config_data.get("show_progress_bar", False),
            normalize_scores=config_data.get("normalize_scores", False),
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
    
    def _get_positions_to_prune(
        self,
        doc_attention_scores: torch.Tensor,
        doc_input_ids: torch.Tensor,
        debug: bool = False,
    ) -> set[int]:
        """
        Identify token positions to prune for a single document based on attention scores.

        Parameters
        ----------
        doc_attention_scores
            Attention scores tensor for this document, shape (seq_len,)
        doc_input_ids
            Input IDs tensor for this document, shape (seq_len,)
        debug
            If True, print debug information about attention scores and pruning decisions

        Returns
        -------
        set[int]
            Set of token positions (indices) to prune in this document
        """
        doc_tokens = doc_input_ids.cpu().tolist()
        attention_scores = doc_attention_scores.cpu()

        if debug:
            print(f"\n  [DEBUG] Input attention scores shape: {attention_scores.shape}")
            print(f"  [DEBUG] Input attention scores: {attention_scores.tolist()}")
            print(f"  [DEBUG] Protected tokens: {self.config.protected_tokens}")

        # Normalize scores if requested
        if self.config.normalize_scores:
            # Apply softmax to normalize (after masking protected tokens if needed)
            # For now, we'll normalize all scores, but we could mask protected tokens first
            attention_scores = torch.softmax(attention_scores, dim=0)
            if debug:
                print(f"  [DEBUG] After softmax normalization: {attention_scores.tolist()}")

        # Score each token occurrence (after protected tokens)
        token_scores: list[tuple[int, float]] = []  # (position, score)

        for pos in range(self.config.protected_tokens, len(doc_tokens)):
            score = attention_scores[pos].item()
            token_scores.append((pos, score))

        if debug:
            print(f"  [DEBUG] Token scores (pos, score) after protected tokens:")
            for pos, score in token_scores[:10]:  # Show first 10
                print(f"    Position {pos}: {score:.4f}")
            if len(token_scores) > 10:
                print(f"    ... and {len(token_scores) - 10} more")

        # Select tokens to prune based on top_k/keep_ratio or threshold
        if self.config.top_k is not None or self.config.keep_ratio is not None:
            # Sort by score (ascending) and take top_k lowest
            sorted_tokens = sorted(token_scores, key=lambda x: x[1])
            if self.config.keep_ratio is not None:
                keep = int(math.ceil(self.config.keep_ratio * len(sorted_tokens)))
                num_to_prune = max(0, len(sorted_tokens) - keep)
            else:
                num_to_prune = self.config.top_k
            # Take min(top_k, len(sorted_tokens)) to handle cases where we have fewer candidates than top_k
            num_to_prune = min(num_to_prune, len(sorted_tokens))
            positions_to_prune = {pos for pos, _ in sorted_tokens[:num_to_prune]}

            if debug:
                print(
                    f"  [DEBUG] Using "
                    f"{'keep_ratio=' + str(self.config.keep_ratio) if self.config.keep_ratio is not None else f'top_k={self.config.top_k}'}, "
                    f"will prune {num_to_prune} tokens"
                )
                print(f"  [DEBUG] Lowest scoring tokens (to be pruned):")
                for pos, score in sorted_tokens[:num_to_prune]:
                    print(f"    Position {pos}: {score:.4f}")
        else:
            # Prune tokens with score < threshold
            positions_to_prune = {
                pos for pos, score in token_scores
                if score < self.config.threshold
            }

            if debug:
                print(f"  [DEBUG] Using threshold={self.config.threshold}")
                print(f"  [DEBUG] Tokens below threshold (to be pruned): {len(positions_to_prune)}")
                pruned_list = [(pos, score) for pos, score in token_scores if score < self.config.threshold]
                for pos, score in pruned_list[:10]:  # Show first 10
                    print(f"    Position {pos}: {score:.4f}")
                if len(pruned_list) > 10:
                    print(f"    ... and {len(pruned_list) - 10} more")

        return positions_to_prune
    
    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply attention-based pruning to embeddings and update artifacts.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain "attention_scores" as a list of tensors.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pruned_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with pruned embeddings.
            Metadata artifacts are passed through unchanged.
        
        Raises
        ------
        ValueError
            If attention_scores artifact is missing
        """
        # Get attention_scores from artifacts
        if "attention_scores" not in artifacts:
            raise ValueError(
                "AttentionPruningStrategy requires 'attention_scores' artifact. "
                "Ensure attention_scores are provided when encoding."
            )

        attention_scores = artifacts["attention_scores"]
        if not isinstance(attention_scores, list):
            raise ValueError("attention_scores artifact must be a list of tensors")

        if len(attention_scores) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(attention_scores)} attention_scores but {len(embeddings)} embeddings"
            )

        if self.debug:
            print(f"\n[DEBUG AttentionPruningStrategy.compress]")
            print(f"  Number of documents: {len(embeddings)}")
            print(f"  Embedding shapes: {[e.shape for e in embeddings]}")
            print(f"  Attention scores shapes: {[a.shape for a in attention_scores]}")
            print(f"  Config: top_k={self.config.top_k}, threshold={self.config.threshold}")
            print(f"  Protected tokens: {self.config.protected_tokens}")
            print(f"  Normalize scores: {self.config.normalize_scores}")
        
        # Validate all attention_scores are tensors
        for idx, doc_attention in enumerate(attention_scores):
            if not isinstance(doc_attention, torch.Tensor):
                raise ValueError(f"attention_scores[{idx}] must be a torch.Tensor, got {type(doc_attention)}")
            if len(doc_attention.shape) != 1:
                raise ValueError(f"attention_scores[{idx}] must be 1D, got shape {doc_attention.shape}")
        
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
        
        # Prune tokens based on attention scores
        pruned_embeddings = []
        updated_attention_scores = []
        updated_input_ids = []
        
        criterion_str = (
            f"top_k={self.config.top_k}"
            if self.config.top_k
            else f"threshold={self.config.threshold}"
            if self.config.threshold is not None
            else f"keep_ratio={self.config.keep_ratio}"
        )
        iterator = tqdm(
            enumerate(zip(embeddings, attention_scores)),
            desc=f"Attention pruning ({criterion_str})",
            total=len(embeddings),
            disable=not self.config.show_progress_bar,
        )
        
        for batch_doc_idx, (doc_embeddings, doc_attention_scores) in iterator:
            # Get input_ids for this document if available
            doc_input_ids = input_ids[batch_doc_idx] if input_ids is not None else None

            if self.debug:
                print(f"\n  [DEBUG] Processing document {batch_doc_idx}")
                print(f"    Embedding shape: {doc_embeddings.shape}")
                print(f"    Attention scores shape: {doc_attention_scores.shape}")
                print(f"    Attention scores (first 10): {doc_attention_scores[:10].tolist()}")

            # Ensure attention scores match embedding length
            if doc_attention_scores.shape[0] != doc_embeddings.shape[0]:
                raise ValueError(
                    f"Document {start_idx + batch_doc_idx}: attention_scores length "
                    f"({doc_attention_scores.shape[0]}) does not match embedding length "
                    f"({doc_embeddings.shape[0]})"
                )

            # Get positions to prune
            if doc_input_ids is not None:
                positions_to_prune = self._get_positions_to_prune(doc_attention_scores, doc_input_ids, debug=self.debug)
            else:
                # If no input_ids, we can still prune based on attention scores alone
                # Create a dummy input_ids tensor for the function
                dummy_input_ids = torch.arange(doc_attention_scores.shape[0], device=doc_attention_scores.device)
                positions_to_prune = self._get_positions_to_prune(doc_attention_scores, dummy_input_ids, debug=self.debug)
            
            # Create mask: keep tokens not in positions_to_prune
            keep_mask = []
            pruned_token_ids = []
            
            for pos in range(doc_embeddings.shape[0]):
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
            
            # Apply mask to embeddings and attention scores
            keep_mask_tensor = torch.tensor(keep_mask, device=doc_embeddings.device, dtype=torch.bool)
            pruned_doc_embeddings = doc_embeddings[keep_mask_tensor]
            pruned_doc_attention = doc_attention_scores[keep_mask_tensor]

            if self.debug:
                print(f"    Positions to prune: {sorted(positions_to_prune)}")
                print(f"    Tokens kept: {keep_mask_tensor.sum().item()} / {len(keep_mask_tensor)}")
                print(f"    Pruned embedding shape: {pruned_doc_embeddings.shape}")

            pruned_embeddings.append(pruned_doc_embeddings)
            updated_attention_scores.append(pruned_doc_attention)
            
            # Update input_ids if available
            if doc_input_ids is not None:
                pruned_doc_tokens = doc_input_ids[keep_mask_tensor]
                updated_input_ids.append(pruned_doc_tokens)
            
            if self.config.track_pruned_tokens:
                if is_parallel_mode:
                    # Parallel mode: set at correct index
                    self._pruned_tokens[start_idx + batch_doc_idx] = pruned_token_ids
                else:
                    # Sequential mode: append
                    self._pruned_tokens.append(pruned_token_ids)
        
        # Update artifacts
        updated_artifacts = {}
        for artifact_name, artifact_value in artifacts.items():
            if artifact_name == "attention_scores":
                # Update attention_scores to match pruned embeddings
                updated_artifacts[artifact_name] = updated_attention_scores
            elif artifact_name == "input_ids" and input_ids is not None:
                # Update input_ids to match pruned embeddings
                updated_artifacts[artifact_name] = updated_input_ids
            elif isinstance(artifact_value, list):
                # Shape-matched artifact: apply same pruning mask
                pruned_artifacts = []
                for doc_idx, artifact_tokens in enumerate(artifact_value):
                    doc_attention_scores = attention_scores[doc_idx]
                    doc_input_ids_for_mask = input_ids[doc_idx] if input_ids is not None else None
                    
                    # Get positions to prune (same logic as above)
                    if doc_input_ids_for_mask is not None:
                        positions_to_prune = self._get_positions_to_prune(doc_attention_scores, doc_input_ids_for_mask)
                    else:
                        dummy_input_ids = torch.arange(doc_attention_scores.shape[0], device=doc_attention_scores.device)
                        positions_to_prune = self._get_positions_to_prune(doc_attention_scores, dummy_input_ids)
                    
                    # Create keep mask
                    keep_mask = []
                    for pos in range(len(artifact_tokens)):
                        if pos < self.config.protected_tokens:
                            keep_mask.append(True)
                        elif pos in positions_to_prune:
                            keep_mask.append(False)
                        else:
                            keep_mask.append(True)
                    
                    # Apply mask to artifact
                    keep_mask_tensor = torch.tensor(keep_mask, device=artifact_tokens.device, dtype=torch.bool)
                    pruned_artifact = artifact_tokens[keep_mask_tensor]
                    pruned_artifacts.append(pruned_artifact)
                
                updated_artifacts[artifact_name] = pruned_artifacts
            else:
                # Metadata artifact: pass through unchanged
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
        Apply attention-based pruning in parallel.
        
        This method parallelizes the per-document pruning operations using the base
        class parallel implementation.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain "attention_scores" as a list of tensors.
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
