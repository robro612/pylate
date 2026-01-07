from .base import *

@dataclass
class PoolingConfig(CompressionStrategyConfigBase):
    pool_factor: int = 1
    protected_tokens: int = 1
    clustering_method: Literal["hierarchical", "spherical"] = "hierarchical"
    show_progress_bar: bool = False
    kmeans_gpu: bool = False  # Enable GPU for fastkmeans (experimental, will fallback to CPU on error)

    def serialize(self) -> dict:
        """
        Serialize this configuration to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "pool_factor": self.pool_factor,
            "protected_tokens": self.protected_tokens,
            "clustering_method": self.clustering_method,
            "show_progress_bar": self.show_progress_bar,
            "kmeans_gpu": self.kmeans_gpu,
        }

    @property
    def strategy_type(self) -> str:
        return "pooling"

class PoolingStrategy(CompressionStrategy):
    """
    Pooling strategy that wraps the hierarchical/spherical pooling logic from ColBERT.

    This strategy pools embeddings by clustering similar token embeddings together
    and averaging them, reducing the number of tokens per document while preserving
    semantic information.

    The hierarchical method uses Ward's linkage clustering on cosine similarity distances.
    The spherical method uses fastkmeans clustering on the embeddings.
    """

    required_artifacts: list[str] = []  # Pooling doesn't require any artifacts

    def __init__(self, config: PoolingConfig):
        """
        Initialize the pooling strategy.

        Parameters
        ----------
        config
            Pooling configuration specifying pool_factor, protected_tokens, and clustering_method
        """
        if config.pool_factor <= 0:
            raise ValueError("`pool_factor` must be a positive integer.")
        if config.protected_tokens < 0:
            raise ValueError("`protected_tokens` must be non-negative.")

        self.config = config

    @property
    def name(self) -> str:
        """Name of this compression strategy."""
        return f"pooling-{self.config.clustering_method}_k-{self.config.pool_factor}_p-{self.config.protected_tokens}"

    @property
    def strategy_type(self) -> str:
        """Strategy type identifier for serialization."""
        return "pooling"
    
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
    def from_dict(cls, data: dict) -> "PoolingStrategy":
        """
        Create a strategy instance from a serialized dictionary.
        
        Parameters
        ----------
        data
            Dictionary containing serialized strategy data
            
        Returns
        -------
        PoolingStrategy
            Deserialized strategy instance
        """
        config_data = data.get("config", {})
        config = PoolingConfig(
            pool_factor=config_data.get("pool_factor", 1),
            protected_tokens=config_data.get("protected_tokens", 1),
            clustering_method=config_data.get("clustering_method", "hierarchical"),
            show_progress_bar=config_data.get("show_progress_bar", False),
            kmeans_gpu=config_data.get("kmeans_gpu", False),
        )
        return cls(config)
    
    def _pool_embeddings_hierarchical(
        self,
        documents_embeddings: list[torch.Tensor],
        pool_factor: int,
        protected_tokens: int,
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """
        Pools the embeddings hierarchically by clustering and averaging them.
        
        This method wraps the exact same logic as ColBERT.pool_embeddings_hierarchical.
        
        Parameters
        ----------
        documents_embeddings
            A list of embeddings for each document.
        pool_factor
            Factor to determine the number of clusters.
        protected_tokens
            Number of tokens to protect from pooling at the start of each document.
        
        Returns
        -------
        tuple[list[torch.Tensor], list[list[int]]]
            A tuple of (pooled_embeddings, cluster_assignments).
            pooled_embeddings: A list of pooled embeddings for each document.
            cluster_assignments: A list of cluster assignment lists, one per document.
                Each assignment list maps original token indices (after protected_tokens) to cluster IDs.
        """
        # Determine device from first embedding (respect original device)
        # Only use CUDA if all embeddings are already on CUDA, otherwise use CPU
        if documents_embeddings:
            first_device = documents_embeddings[0].device
            # Use CUDA only if CUDA is available AND all embeddings are already on CUDA
            if torch.cuda.is_available() and first_device.type == "cuda":
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        
        pooled_embeddings = []
        cluster_assignments = []
        
        iterator = tqdm(
            documents_embeddings,
            desc=f"Hierarchical pooling (factor={pool_factor})",
            disable=not self.config.show_progress_bar,
            leave=False,
        )
        
        for document_embeddings in iterator:
            document_embeddings = document_embeddings.to(device=device)
            
            # Separate protected tokens from the rest
            # Ensure protected_tokens doesn't exceed document length to avoid CUDA asserts
            num_doc_tokens = document_embeddings.shape[0]
            actual_protected = min(protected_tokens, num_doc_tokens)
            protected_embeddings = document_embeddings[:actual_protected]
            embeddings_to_pool = document_embeddings[actual_protected:]
            
            num_embeddings = len(embeddings_to_pool)
            
            # If no embeddings to pool, just return protected embeddings
            if num_embeddings == 0:
                pooled_embeddings.append(protected_embeddings)
                cluster_assignments.append([])
                continue

            # Compute cosine similarity and convert to distance matrix
            # Cast to float32 for torch.mm compatibility (BFloat16 not supported on all platforms)
            embeddings_float32 = embeddings_to_pool.float()
            cosine_similarities = torch.mm(
                input=embeddings_float32, mat2=embeddings_float32.t()
            )
            distance_matrix = 1 - cosine_similarities.cpu().numpy()
            
            # Perform hierarchical clustering using Ward's method
            clusters = hierarchy.linkage(distance_matrix, method="ward")
            
            # Determine the number of clusters based on pool_factor
            num_clusters = max(num_embeddings // pool_factor, 1)
            cluster_labels = hierarchy.fcluster(
                clusters, t=num_clusters, criterion="maxclust"
            )
            
            # Store cluster assignments for artifact mapping
            cluster_assignments.append(cluster_labels.tolist())
            
            # Pool embeddings within each cluster
            pooled_document_embeddings = []
            for cluster_id in range(1, num_clusters + 1):
                cluster_indices = torch.where(
                    condition=torch.tensor(
                        data=cluster_labels == cluster_id, device=device
                    )
                )[0]
                if cluster_indices.numel() > 0:
                    cluster_embedding = embeddings_to_pool[cluster_indices].mean(dim=0)
                    pooled_document_embeddings.append(cluster_embedding)
            
            # Re-append protected embeddings
            pooled_document_embeddings.extend(protected_embeddings)
            pooled_embeddings.append(torch.stack(tensors=pooled_document_embeddings))
        
        return pooled_embeddings, cluster_assignments
    
    def _pool_embeddings_spherical(
        self,
        documents_embeddings: list[torch.Tensor],
        pool_factor: int,
        protected_tokens: int,
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """
        Pools the embeddings using spherical clustering via fastkmeans.
        
        This method uses fastkmeans to perform k-means clustering on the embeddings,
        then averages embeddings within each cluster to create pooled representations.
        
        Parameters
        ----------
        documents_embeddings
            A list of embeddings for each document.
        pool_factor
            Factor to determine the number of clusters.
        protected_tokens
            Number of tokens to protect from pooling at the start of each document.
        
        Returns
        -------
        tuple[list[torch.Tensor], list[list[int]]]
            A tuple of (pooled_embeddings, cluster_assignments).
            pooled_embeddings: A list of pooled embeddings for each document.
            cluster_assignments: A list of cluster assignment lists, one per document.
                Each assignment list maps original token indices (after protected_tokens) to cluster IDs.
        
        Raises
        ------
        ImportError
            If fastkmeans is not installed
        """
        if fastkmeans is None:
            raise ImportError(
                "fastkmeans is required for spherical clustering. "
                "Install it with: pip install fastkmeans"
            )
        
        # Determine device from first embedding (respect original device)
        # Only use CUDA if all embeddings are already on CUDA, otherwise use CPU
        if documents_embeddings:
            first_device = documents_embeddings[0].device
            # Use CUDA only if CUDA is available AND all embeddings are already on CUDA
            if torch.cuda.is_available() and first_device.type == "cuda":
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        
        pooled_embeddings = []
        cluster_assignments = []
        
        iterator = tqdm(
            documents_embeddings,
            desc=f"Spherical pooling (factor={pool_factor})",
            disable=not self.config.show_progress_bar,
            leave=False,
        )
        
        for document_embeddings in iterator:
            document_embeddings = document_embeddings.to(device=device)
            
            # Separate protected tokens from the rest
            # Ensure protected_tokens doesn't exceed document length to avoid CUDA asserts
            num_doc_tokens = document_embeddings.shape[0]
            actual_protected = min(protected_tokens, num_doc_tokens)
            protected_embeddings = document_embeddings[:actual_protected]
            embeddings_to_pool = document_embeddings[actual_protected:]
            
            num_embeddings = len(embeddings_to_pool)
            
            # If no embeddings to pool, just return protected embeddings
            if num_embeddings == 0:
                if protected_tokens > 0:
                    pooled_embeddings.append(protected_embeddings)
                else:
                    # Empty document case
                    pooled_embeddings.append(torch.empty((0, document_embeddings.shape[1]), device=device))
                cluster_assignments.append([])
                continue
            
            # Ensure we have at least one embedding dimension
            if document_embeddings.shape[1] == 0:
                pooled_embeddings.append(document_embeddings)
                cluster_assignments.append([])
                continue
            
            # Determine the number of clusters based on pool_factor
            num_clusters = max(num_embeddings // pool_factor, 1)
            
            # If we have fewer embeddings than clusters, just use all embeddings
            if num_clusters >= num_embeddings:
                pooled_embeddings.append(document_embeddings)
                cluster_assignments.append(list(range(1, num_embeddings + 1)))
                continue
            
            # Normalize embeddings for spherical k-means (cosine similarity)
            # Spherical k-means works on unit vectors
            embeddings_normalized = torch.nn.functional.normalize(
                embeddings_to_pool, p=2, dim=1
            )
            
            # Convert to numpy for fastkmeans (it expects numpy arrays)
            embeddings_np = embeddings_normalized.cpu().float().numpy()
            embedding_dim = embeddings_np.shape[1]

            # Determine if we should try GPU
            use_gpu = self.config.kmeans_gpu and torch.cuda.is_available()

            # Initialize and train fastkmeans with GPU fallback
            cluster_labels_np = None
            if use_gpu:
                try:
                    # Sync CUDA before clustering to ensure clean state
                    torch.cuda.synchronize()

                    kmeans = fastkmeans.FastKMeans(
                        embedding_dim,
                        num_clusters,
                        niter=10,
                        gpu=True,
                        verbose=False,
                        seed=42,
                    )
                    kmeans.train(embeddings_np)
                    cluster_labels_np = kmeans.predict(embeddings_np)
                except RuntimeError as e:
                    # GPU failed, will fallback to CPU below
                    if "CUDA" in str(e) or "cuda" in str(e).lower():
                        import warnings
                        warnings.warn(
                            f"fastkmeans GPU failed with CUDA error, falling back to CPU: {e}",
                            RuntimeWarning,
                        )
                        cluster_labels_np = None
                    else:
                        raise

            # CPU fallback (or if GPU was not requested)
            if cluster_labels_np is None:
                kmeans = fastkmeans.FastKMeans(
                    embedding_dim,
                    num_clusters,
                    niter=10,
                    gpu=False,
                    verbose=False,
                    seed=42,
                )
                kmeans.train(embeddings_np)
                cluster_labels_np = kmeans.predict(embeddings_np)
            
            # Convert cluster labels to torch tensor for indexing
            cluster_labels_tensor = torch.from_numpy(cluster_labels_np).to(device=device)
            
            # Convert cluster labels to list (for storage)
            cluster_labels_list = cluster_labels_np.tolist()
            
            # Store cluster assignments (convert to 1-indexed to match hierarchical)
            # fastkmeans uses 0-indexed, but we need 1-indexed to match hierarchical format
            cluster_assignments.append([label + 1 for label in cluster_labels_list])
            
            # Pool embeddings within each cluster by averaging
            pooled_document_embeddings = []
            for cluster_id in range(num_clusters):
                # Find indices of embeddings belonging to this cluster
                cluster_indices = torch.where(cluster_labels_tensor == cluster_id)[0]
                if cluster_indices.numel() > 0:
                    # Average the embeddings in this cluster
                    cluster_embedding = embeddings_to_pool[cluster_indices].mean(dim=0)
                    pooled_document_embeddings.append(cluster_embedding)
            
            # Re-append protected embeddings
            pooled_document_embeddings.extend(protected_embeddings)
            pooled_embeddings.append(torch.stack(tensors=pooled_document_embeddings))
        
        return pooled_embeddings, cluster_assignments
    
    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply pooling compression to embeddings and update artifacts to maintain 1:1 mapping.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Shape-matched artifacts will be updated to match
            the pooled embeddings. Metadata artifacts are passed through unchanged.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pooled_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with pooled embeddings.
            Metadata artifacts are passed through unchanged.
        """
        # Skip pooling if pool_factor is 1 (no pooling)
        if self.config.pool_factor == 1:
            return embeddings, artifacts
        
        # Apply pooling based on clustering method
        if self.config.clustering_method == "hierarchical":
            pooled_embeddings, cluster_assignments = self._pool_embeddings_hierarchical(
                documents_embeddings=embeddings,
                pool_factor=self.config.pool_factor,
                protected_tokens=self.config.protected_tokens,
            )
        elif self.config.clustering_method == "spherical":
            pooled_embeddings, cluster_assignments = self._pool_embeddings_spherical(
                documents_embeddings=embeddings,
                pool_factor=self.config.pool_factor,
                protected_tokens=self.config.protected_tokens,
            )
        else:
            raise ValueError(
                f"Unknown clustering method: {self.config.clustering_method}. "
                f"Must be 'hierarchical' or 'spherical'."
            )
        
        # Update shape-matched artifacts to match pooled embeddings
        # Use cluster assignments to map original tokens to pooled tokens
        updated_artifacts = {}
        for artifact_name, artifact_value in artifacts.items():
            if isinstance(artifact_value, list):
                # Shape-matched artifact: need to pool it using cluster assignments
                pooled_artifacts = []
                for doc_idx, artifact_tokens in enumerate(artifact_value):
                    doc_embeddings = embeddings[doc_idx]
                    pooled_doc_embeddings = pooled_embeddings[doc_idx]
                    
                    # Protected tokens map to themselves
                    protected_artifact_tokens = artifact_tokens[:self.config.protected_tokens]
                    
                    # Convert artifact_tokens to list if it's a tensor
                    if isinstance(artifact_tokens, torch.Tensor):
                        artifact_tokens_list = artifact_tokens.tolist()
                    else:
                        artifact_tokens_list = list(artifact_tokens)
                    
                    # Map pooled tokens using cluster assignments
                    pooled_artifact_tokens_list = list(protected_artifact_tokens)
                    
                    # Use cluster assignments to select representative tokens
                    doc_cluster_labels = cluster_assignments[doc_idx]
                    num_clusters = max(len(doc_cluster_labels) // self.config.pool_factor, 1)
                    
                    # For each cluster, select the first token as representative
                    for cluster_id in range(1, num_clusters + 1):
                        cluster_token_indices = [
                            i for i, label in enumerate(doc_cluster_labels)
                            if label == cluster_id
                        ]
                        if cluster_token_indices:
                            # Use the first token in the cluster as representative
                            original_idx = cluster_token_indices[0] + self.config.protected_tokens
                            if original_idx < len(artifact_tokens_list):
                                pooled_artifact_tokens_list.append(artifact_tokens_list[original_idx])
                    
                    # Ensure we have the right number of tokens (should match pooled embeddings)
                    while len(pooled_artifact_tokens_list) < len(pooled_doc_embeddings):
                        # Pad with last token if needed (shouldn't happen, but safety check)
                        if artifact_tokens_list:
                            pooled_artifact_tokens_list.append(artifact_tokens_list[-1])
                        else:
                            break
                    
                    pooled_artifact_tokens_list = pooled_artifact_tokens_list[:len(pooled_doc_embeddings)]
                    
                    # Convert back to tensor if original was tensor
                    if isinstance(artifact_value[doc_idx], torch.Tensor):
                        pooled_artifacts.append(torch.tensor(pooled_artifact_tokens_list, device=artifact_value[doc_idx].device, dtype=artifact_value[doc_idx].dtype))
                    else:
                        pooled_artifacts.append(pooled_artifact_tokens_list)
                
                updated_artifacts[artifact_name] = pooled_artifacts
            else:
                # Metadata artifact: pass through unchanged
                updated_artifacts[artifact_name] = artifact_value
        
        return pooled_embeddings, updated_artifacts


