"""ProxyAttentionColBERT: ColBERT with learned proxy query tokens for attention-based compression."""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Iterable, Optional, override

import torch
import torch.nn.functional as F
from torch import nn

from ..hf_hub.model_card import PylateModelCardData
from ..scores import SimilarityFunction
from .colbert import ColBERT

logger = logging.getLogger(__name__)


class ProxyEmbeddingsModule(nn.Module):
    """
    A wrapper module to hold proxy embeddings as learnable parameters.

    This is a simple container that holds the proxy embedding weights as
    nn.Parameter so they can be trained and saved with the model.

    Parameters
    ----------
    num_proxy_tokens : int
        Number of proxy query tokens
    hidden_size : int
        Dimension of each proxy token embedding
    init_std : float
        Standard deviation for Gaussian initialization
    """

    def __init__(
        self,
        num_proxy_tokens: int,
        hidden_size: int,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.num_proxy_tokens = num_proxy_tokens
        self.hidden_size = hidden_size
        self.init_std = init_std

        # Create weight as a Parameter (not Embedding to avoid issues with Sequential)
        self.weight = nn.Parameter(torch.empty(num_proxy_tokens, hidden_size))

        # Initialize with random Gaussian
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, features: dict) -> dict:
        """
        Dummy forward method to satisfy SentenceTransformer's module iteration.

        This module is not meant to be called in the forward pass - it just stores
        the proxy embeddings. The actual usage is via get_embeddings().
        """
        return features

    def get_embeddings(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Get proxy embeddings expanded for a batch.

        Returns
        -------
        torch.Tensor
            Shape (batch_size, num_proxy_tokens, hidden_size)
        """
        # Get embeddings and expand for batch
        proxy_embeds = self.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return proxy_embeds.to(device=device, dtype=dtype)

    def save(self, output_path: str, safe_serialization: bool = True):
        """Save proxy embeddings to disk."""
        os.makedirs(output_path, exist_ok=True)

        # Save config
        config = {
            "num_proxy_tokens": self.num_proxy_tokens,
            "hidden_size": self.hidden_size,
            "init_std": self.init_std,
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save weights
        if safe_serialization:
            from safetensors.torch import save_file
            save_file(
                {"weight": self.weight.data},
                os.path.join(output_path, "model.safetensors")
            )
        else:
            torch.save(
                {"weight": self.weight.data},
                os.path.join(output_path, "pytorch_model.bin")
            )

    @staticmethod
    def load(input_path: str) -> "ProxyEmbeddingsModule":
        """Load proxy embeddings from disk."""
        # Load config
        with open(os.path.join(input_path, "config.json"), "r") as f:
            config = json.load(f)

        # Create module
        module = ProxyEmbeddingsModule(
            num_proxy_tokens=config["num_proxy_tokens"],
            hidden_size=config["hidden_size"],
            init_std=config.get("init_std", 0.02),
        )

        # Load weights
        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(input_path, "model.safetensors"))
        else:
            state_dict = torch.load(
                os.path.join(input_path, "pytorch_model.bin"),
                map_location="cpu"
            )

        module.weight.data.copy_(state_dict["weight"])
        return module


class ProxyAttentionColBERT(ColBERT):
    """
    ColBERT with learned proxy query tokens for attention-based document compression.

    This model extends ColBERT by using learned proxy query tokens to compute
    saliency scores over document tokens. The top-k most salient tokens are
    selected as centroids, and cluster-based pooling is applied to compress
    the document representation.

    Key Features:
    - Proxy tokens are appended to document sequences (not queries)
    - Last transformer layer uses eager attention to output attention weights
    - Saliency is computed from attention weights (proxy → document tokens)
    - Hard top-k selection picks centroids based on saliency
    - Cluster pooling aggregates tokens around centroids

    Parameters
    ----------
    model_name_or_path : str | None
        Model name or path (same as ColBERT)
    num_proxy_tokens : int
        Number of proxy query tokens (n_ψ). Default: 32
    num_select_tokens : int
        Number of tokens to select/output (m). Default: 32
    proxy_tau : float
        Temperature for attention softmax. Default: 1.0
    use_cluster_pooling : bool
        Whether to pool tokens around selected centroids. Default: True
    cluster_centroid_weight : float
        Extra weight for centroids in cluster pooling. Default: 1.0
    use_attn_weight_cluster_pooling : bool
        Use saliency scores as weights in cluster pooling. Default: False
    proxy_init_std : float
        Standard deviation for proxy embedding initialization. Default: 0.02
    **kwargs
        All other ColBERT parameters
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        # Proxy attention parameters
        num_proxy_tokens: int = 32,
        num_select_tokens: int = 32,
        proxy_tau: float = 1.0,
        use_cluster_pooling: bool = True,
        cluster_centroid_weight: float = 1.0,
        use_attn_weight_cluster_pooling: bool = True,
        proxy_init_std: float = 0.02,
        normalize_embeddings: bool = True,
        # ColBERT parameters
        embedding_size: int | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        query_length: int | None = None,
        document_length: int | None = None,
        attend_to_expansion_tokens: bool = False,
        trust_remote_code: bool = False,
        skiplist_words: Iterable[str] | None = None,
        do_query_expansion: bool = False,
        model_card_data: PylateModelCardData | None = None,
        similarity_fn_name: SimilarityFunction | str = SimilarityFunction.MAXSIM,
        **kwargs,
    ):
        # Initialize base ColBERT
        super().__init__(
            model_name_or_path=model_name_or_path,
            embedding_size=embedding_size,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            query_length=query_length,
            document_length=document_length,
            attend_to_expansion_tokens=attend_to_expansion_tokens,
            trust_remote_code=trust_remote_code,
            skiplist_words=skiplist_words,
            do_query_expansion=do_query_expansion,
            model_card_data=model_card_data,
            similarity_fn_name=similarity_fn_name,
            **kwargs,
        )

        # Proxy attention configuration
        self.num_proxy_tokens = num_proxy_tokens
        self.num_select_tokens = num_select_tokens
        self.proxy_tau = proxy_tau
        self.use_cluster_pooling = use_cluster_pooling
        self.cluster_centroid_weight = cluster_centroid_weight
        self.use_attn_weight_cluster_pooling = use_attn_weight_cluster_pooling
        self.proxy_init_std = proxy_init_std
        self.normalize_embeddings = normalize_embeddings

        # Get hidden size from transformer
        transformer = self[0]
        auto_model = transformer.auto_model
        hidden_size = auto_model.config.hidden_size

        # Initialize proxy embeddings (stored separately, not in Sequential modules)
        self._proxy_embeddings = ProxyEmbeddingsModule(
            num_proxy_tokens=num_proxy_tokens,
            hidden_size=hidden_size,
            init_std=proxy_init_std,
        )

        # Enable eager attention for the last layer (required for attention weights)
        self._enable_eager_for_last_layer()

        logger.info(
            f"Initialized ProxyAttentionColBERT with {num_proxy_tokens} proxy tokens, "
            f"selecting {num_select_tokens} tokens, cluster_pooling={use_cluster_pooling}"
        )

    def _enable_eager_for_last_layer(self):
        """
        Set the last transformer layer to use eager attention implementation.

        This is required because output_attentions=True only works with eager attention,
        not with flash_attention_2 or sdpa implementations.
        """
        transformer = self[0]
        auto_model = transformer.auto_model

        # Find the last transformer layer based on model architecture
        last_layer = None
        if hasattr(auto_model, 'layers'):
            # ModernBERT style (list of layers)
            last_layer = auto_model.layers[-1]
        elif hasattr(auto_model, 'encoder') and hasattr(auto_model.encoder, 'layer'):
            # BERT/RoBERTa style
            last_layer = auto_model.encoder.layer[-1]
        elif hasattr(auto_model, 'transformer') and hasattr(auto_model.transformer, 'layer'):
            # Some other architectures
            last_layer = auto_model.transformer.layer[-1]
        else:
            raise ValueError(
                f"Cannot find transformer layers in model architecture: {type(auto_model)}. "
                "Please check the model structure and add support for this architecture."
            )

        # Find the attention module within the layer
        attn_module = None
        for name in ['attn', 'attention', 'self_attn', 'self']:
            if hasattr(last_layer, name):
                attn_module = getattr(last_layer, name)
                break

        if attn_module is None:
            raise ValueError(
                f"Cannot find attention module in layer: {type(last_layer)}. "
                f"Available attributes: {dir(last_layer)}"
            )

        # Set attention implementation to eager
        # Different models store this differently
        if hasattr(attn_module, 'config'):
            # Copy config and modify
            new_config = copy.deepcopy(attn_module.config)
            new_config._attn_implementation = 'eager'
            attn_module.config = new_config
            logger.debug("Set last layer attention to eager via config")
        elif hasattr(attn_module, '_attn_implementation'):
            attn_module._attn_implementation = 'eager'
            logger.debug("Set last layer _attn_implementation to eager")
        else:
            # Try setting on the layer itself
            if hasattr(last_layer, 'config'):
                new_config = copy.deepcopy(last_layer.config)
                new_config._attn_implementation = 'eager'
                last_layer.config = new_config
                logger.debug("Set last layer attention to eager via layer config")
            else:
                logger.warning(
                    "Could not set attention implementation to eager. "
                    "output_attentions=True may not work correctly."
                )

    def _get_embedding_layer(self, auto_model) -> nn.Module:
        """Get the word embedding layer from the model."""
        if hasattr(auto_model, 'embeddings'):
            # BERT/ModernBERT style
            if hasattr(auto_model.embeddings, 'word_embeddings'):
                return auto_model.embeddings.word_embeddings
            elif hasattr(auto_model.embeddings, 'tok_embeddings'):
                return auto_model.embeddings.tok_embeddings
            else:
                return auto_model.embeddings
        elif hasattr(auto_model, 'embed_tokens'):
            return auto_model.embed_tokens
        elif hasattr(auto_model, 'wte'):
            # GPT-2 style
            return auto_model.wte
        else:
            raise ValueError(f"Cannot find embedding layer in model: {type(auto_model)}")

    def _append_proxy_tokens(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append proxy token embeddings to document embeddings.

        Parameters
        ----------
        input_embeds : torch.Tensor
            Document token embeddings (batch, seq_len, hidden_dim)
        attention_mask : torch.Tensor
            Attention mask for document tokens (batch, seq_len)

        Returns
        -------
        combined_embeds : torch.Tensor
            Combined embeddings with proxy tokens appended (batch, seq_len + n_proxy, hidden_dim)
        combined_mask : torch.Tensor
            Extended attention mask (batch, seq_len + n_proxy)
        """
        batch_size, seq_len, hidden_dim = input_embeds.shape
        device = input_embeds.device
        dtype = input_embeds.dtype

        # Get proxy embeddings
        proxy_embeds = self._proxy_embeddings.get_embeddings(batch_size, device, dtype)

        # Concatenate: [doc_tokens, proxy_tokens]
        combined_embeds = torch.cat([input_embeds, proxy_embeds], dim=1)

        # Extend attention mask (all proxy tokens are valid)
        proxy_mask = torch.ones(
            batch_size, self.num_proxy_tokens,
            device=device, dtype=attention_mask.dtype
        )
        combined_mask = torch.cat([attention_mask, proxy_mask], dim=1)

        return combined_embeds, combined_mask

    def _create_doc_proxy_masks(
        self,
        attention_mask: torch.Tensor,
        total_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create separate masks for document tokens and proxy tokens.

        Parameters
        ----------
        attention_mask : torch.Tensor
            Combined attention mask (batch, total_seq_len)
        total_seq_len : int
            Total sequence length (doc_len + n_proxy)

        Returns
        -------
        doc_mask : torch.Tensor
            Mask for document tokens (batch, total_seq_len) - 1s for doc positions
        proxy_mask : torch.Tensor
            Mask for proxy tokens (batch, total_seq_len) - 1s for proxy positions
        """
        batch_size = attention_mask.shape[0]
        device = attention_mask.device
        dtype = attention_mask.dtype

        # Document tokens are the first (total_seq_len - n_proxy) positions
        doc_len = total_seq_len - self.num_proxy_tokens

        # Create doc mask: valid document positions only
        doc_mask = torch.zeros(batch_size, total_seq_len, device=device, dtype=dtype)
        doc_mask[:, :doc_len] = attention_mask[:, :doc_len]

        # Create proxy mask: proxy positions only (always valid)
        proxy_mask = torch.zeros(batch_size, total_seq_len, device=device, dtype=dtype)
        proxy_mask[:, doc_len:] = 1.0

        return doc_mask, proxy_mask

    def _compute_saliency_from_attention(
        self,
        attention_weights: torch.Tensor,
        doc_mask: torch.Tensor,
        proxy_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute saliency scores from attention weights.

        Saliency is computed as the mean attention that proxy tokens pay to each
        document token, normalized with softmax over document positions.

        Parameters
        ----------
        attention_weights : torch.Tensor
            Attention weights from last layer (batch, num_heads, seq_len, seq_len)
        doc_mask : torch.Tensor
            Mask for document tokens (batch, seq_len)
        proxy_mask : torch.Tensor
            Mask for proxy tokens (batch, seq_len)

        Returns
        -------
        saliency : torch.Tensor
            Saliency scores for each position (batch, seq_len)
            Only document positions have valid saliency values
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        device = attention_weights.device
        dtype = attention_weights.dtype

        # Average over heads
        attn_avg = attention_weights.mean(dim=1)  # (batch, seq_len, seq_len)

        # Get proxy token indices
        proxy_indices = proxy_mask.bool()  # (batch, seq_len)
        doc_indices = doc_mask.bool()  # (batch, seq_len)

        # Extract attention from proxy tokens to all tokens
        # For each batch, select rows corresponding to proxy tokens
        # Result: (batch, n_proxy, seq_len)
        n_proxy = self.num_proxy_tokens
        proxy_attention = attn_avg[:, -n_proxy:, :]  # Proxy tokens are at the end

        # Mask out non-document positions and apply softmax with temperature
        # Create mask for document positions: (batch, 1, seq_len)
        doc_mask_expanded = doc_mask.unsqueeze(1)  # (batch, 1, seq_len)

        # Apply mask and temperature
        proxy_attention = proxy_attention / self.proxy_tau
        proxy_attention = proxy_attention.masked_fill(~doc_mask_expanded.bool(), float('-inf'))

        # Softmax over document positions
        proxy_attention = F.softmax(proxy_attention, dim=-1)  # (batch, n_proxy, seq_len)

        # Handle NaN from all-masked rows (shouldn't happen if doc_mask is valid)
        proxy_attention = torch.nan_to_num(proxy_attention, nan=0.0)

        # Average over proxy tokens to get final saliency
        saliency = proxy_attention.mean(dim=1)  # (batch, seq_len)

        return saliency

    def _hard_select(
        self,
        doc_hidden_states: torch.Tensor,
        saliency_scores: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hard top-k selection with optional cluster pooling.

        Select the top m document tokens based on saliency scores.
        If cluster pooling is enabled, use selected tokens as centroids
        and average all tokens within each cluster.

        Parameters
        ----------
        doc_hidden_states : torch.Tensor
            Document token embeddings (batch, seq_len, hidden_dim)
        saliency_scores : torch.Tensor
            Saliency scores for each position (batch, seq_len)
        doc_mask : torch.Tensor
            Mask for valid document positions (batch, seq_len)

        Returns
        -------
        selected_embeddings : torch.Tensor
            Selected/pooled embeddings (batch, m, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = doc_hidden_states.shape

        # Only consider document positions (not proxy tokens)
        # Mask out proxy positions from saliency scores
        masked_saliency = saliency_scores.clone()
        masked_saliency = masked_saliency.masked_fill(~doc_mask.bool(), float('-inf'))

        # Get number of valid document tokens
        n_doc = doc_mask.sum(dim=-1).min().int().item()  # Minimum across batch
        m = min(self.num_select_tokens, n_doc)

        # Get top-k indices based on saliency scores
        _, topk_indices = torch.topk(masked_saliency, k=m, dim=-1, largest=True, sorted=False)

        # Gather the selected embeddings (centroids)
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        centroid_embeddings = torch.gather(doc_hidden_states, dim=1, index=expanded_indices)

        if self.use_cluster_pooling:
            weights = saliency_scores if self.use_attn_weight_cluster_pooling else None
            return self._cluster_pool(
                doc_hidden_states, centroid_embeddings, topk_indices, doc_mask, weights
            )
        else:
            return centroid_embeddings

    def _cluster_pool(
        self,
        doc_hidden_states: torch.Tensor,
        centroids: torch.Tensor,
        centroid_indices: torch.Tensor,
        doc_mask: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cluster pooling: assign each document token to its nearest centroid
        and compute weighted average within each cluster.

        Parameters
        ----------
        doc_hidden_states : torch.Tensor
            All document token embeddings (batch, seq_len, hidden_dim)
        centroids : torch.Tensor
            Selected centroid embeddings (batch, m, hidden_dim)
        centroid_indices : torch.Tensor
            Indices of centroid tokens (batch, m)
        doc_mask : torch.Tensor
            Mask for valid document positions (batch, seq_len)
        weights : Optional[torch.Tensor]
            Optional weights for each token (batch, seq_len), e.g., saliency scores

        Returns
        -------
        pooled_embeddings : torch.Tensor
            Cluster-averaged embeddings (batch, m, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = doc_hidden_states.shape
        _, m, _ = centroids.shape

        # Normalize for cosine similarity
        doc_norm = F.normalize(doc_hidden_states, p=2, dim=-1)
        centroid_norm = F.normalize(centroids, p=2, dim=-1)

        # Compute similarity between all tokens and centroids
        # (batch, seq_len, hidden_dim) @ (batch, hidden_dim, m) -> (batch, seq_len, m)
        similarity = torch.bmm(doc_norm, centroid_norm.transpose(1, 2))

        # Hard assignment: each token to nearest centroid
        assignments = similarity.argmax(dim=-1)  # (batch, seq_len)

        # Create weight tensor
        if weights is None:
            weights = torch.ones(batch_size, seq_len, device=doc_hidden_states.device, dtype=doc_hidden_states.dtype)
        else:
            # Ensure weights match dtype
            weights = weights.to(dtype=doc_hidden_states.dtype)

        # Apply document mask (only doc tokens participate in pooling)
        weights = weights * doc_mask.to(dtype=doc_hidden_states.dtype)

        # Apply extra weight to centroid tokens if configured
        if self.cluster_centroid_weight != 1.0:
            centroid_mask = torch.zeros(batch_size, seq_len, device=doc_hidden_states.device, dtype=doc_hidden_states.dtype)
            centroid_mask.scatter_(1, centroid_indices, 1.0)
            weight_factor = torch.tensor(self.cluster_centroid_weight - 1.0, dtype=doc_hidden_states.dtype, device=doc_hidden_states.device)
            weights = weights + weight_factor * centroid_mask * doc_mask.to(dtype=doc_hidden_states.dtype)

        # Create weighted one-hot assignment matrix: (batch, seq_len, m)
        one_hot = F.one_hot(assignments, num_classes=m).to(doc_hidden_states.dtype)
        weighted_one_hot = one_hot * weights.unsqueeze(-1)

        # Sum of weights per cluster
        min_val = torch.tensor(1e-8, dtype=doc_hidden_states.dtype, device=doc_hidden_states.device)
        cluster_weight_sums = weighted_one_hot.sum(dim=1).clamp(min=min_val)  # (batch, m)

        # Weighted sum of tokens within each cluster
        # (batch, m, seq_len) @ (batch, seq_len, hidden_dim) -> (batch, m, hidden_dim)
        cluster_sums = torch.bmm(weighted_one_hot.transpose(1, 2), doc_hidden_states)

        # Weighted average
        pooled_embeddings = cluster_sums / cluster_weight_sums.unsqueeze(-1)

        return pooled_embeddings

    def _encode_document_with_proxy_attention(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Encode document with proxy attention selection.

        This is the core document encoding method that:
        1. Gets input embeddings from the transformer
        2. Appends proxy token embeddings
        3. Runs through transformer with output_attentions=True
        4. Computes saliency from attention weights
        5. Performs hard selection with optional cluster pooling

        Parameters
        ----------
        features : dict
            Input features with 'input_ids', 'attention_mask', etc.

        Returns
        -------
        dict
            Features with 'token_embeddings' replaced by selected embeddings
        """
        transformer = self[0]
        auto_model = transformer.auto_model

        input_ids = features['input_ids']
        attention_mask = features['attention_mask']

        batch_size = input_ids.shape[0]

        # Get word embeddings
        embedding_layer = self._get_embedding_layer(auto_model)
        input_embeds = embedding_layer(input_ids)

        # Append proxy token embeddings
        combined_embeds, combined_mask = self._append_proxy_tokens(input_embeds, attention_mask)

        # Run through transformer with output_attentions for last layer
        outputs = auto_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )

        last_hidden_state = outputs.last_hidden_state
        last_attention_weights = outputs.attentions[-1]  # Last layer attention

        total_seq_len = last_hidden_state.shape[1]

        # Create masks for doc and proxy tokens
        doc_mask, proxy_mask = self._create_doc_proxy_masks(combined_mask, total_seq_len)

        # Compute saliency from attention weights
        saliency_scores = self._compute_saliency_from_attention(
            last_attention_weights, doc_mask, proxy_mask
        )

        # Hard selection with optional cluster pooling
        selected_embeddings = self._hard_select(
            last_hidden_state, saliency_scores, doc_mask
        )

        # Update features with selected embeddings
        features['token_embeddings'] = selected_embeddings
        features['attention_mask'] = torch.ones(
            batch_size, selected_embeddings.shape[1],
            device=selected_embeddings.device, dtype=attention_mask.dtype
        )

        return features

    @override
    def forward(
        self,
        features: dict[str, torch.Tensor],
        is_query: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with proxy attention for documents.

        Queries are encoded using standard ColBERT encoding.
        Documents are encoded with proxy attention selection.

        Parameters
        ----------
        features : dict
            Input features from tokenizer
        is_query : bool, optional
            Whether this is a query (True) or document (False)

        Returns
        -------
        dict
            Features with 'token_embeddings' and 'sentence_embedding'
        """
        if is_query is None:
            is_query = True  # Default to query mode

        if is_query:
            # Standard ColBERT encoding for queries
            return super().forward(features, is_query=True)
        else:
            # Document encoding with proxy attention selection
            features_with_selection = self._encode_document_with_proxy_attention(features)

            # Get selected embeddings
            token_embeddings = features_with_selection['token_embeddings']

            # Apply dense layer if present
            for idx in range(1, len(self)):
                module = self[idx]
                # Check if it's a dense layer
                if hasattr(module, 'linear'):
                    token_embeddings = module({"token_embeddings": token_embeddings})["token_embeddings"]

            # Normalize if configured
            if self.normalize_embeddings:
                token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)

            # Create output
            features['token_embeddings'] = token_embeddings
            features['sentence_embedding'] = token_embeddings.mean(dim=1)
            features['attention_mask'] = features_with_selection['attention_mask']

            return features

    @override
    def save(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list | None = None,
        safe_serialization: bool = True,
    ) -> None:
        """
        Save model including proxy embeddings.

        Parameters
        ----------
        path : str
            Output directory path
        model_name : str, optional
            Model name for the model card
        create_model_card : bool
            Whether to create a model card
        train_datasets : list, optional
            Training datasets for the model card
        safe_serialization : bool
            Whether to use safetensors format
        """
        # Call parent save
        super().save(
            path=path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )

        # Save proxy embeddings
        proxy_path = os.path.join(path, "proxy_embeddings")
        self._proxy_embeddings.save(proxy_path, safe_serialization=safe_serialization)

        # Update config with proxy attention parameters
        config_path = os.path.join(path, "config_sentence_transformers.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        config["model_type"] = "ProxyAttentionColBERT"
        config["num_proxy_tokens"] = self.num_proxy_tokens
        config["num_select_tokens"] = self.num_select_tokens
        config["proxy_tau"] = self.proxy_tau
        config["use_cluster_pooling"] = self.use_cluster_pooling
        config["cluster_centroid_weight"] = self.cluster_centroid_weight
        config["use_attn_weight_cluster_pooling"] = self.use_attn_weight_cluster_pooling
        config["proxy_init_std"] = self.proxy_init_std

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved ProxyAttentionColBERT to {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> "ProxyAttentionColBERT":
        """
        Load a ProxyAttentionColBERT model from disk.

        Parameters
        ----------
        path : str
            Path to the saved model directory
        **kwargs
            Additional arguments passed to __init__

        Returns
        -------
        ProxyAttentionColBERT
            Loaded model
        """
        # Load config
        config_path = os.path.join(path, "config_sentence_transformers.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Extract proxy attention parameters
        proxy_params = {
            "num_proxy_tokens": config.get("num_proxy_tokens", 32),
            "num_select_tokens": config.get("num_select_tokens", 32),
            "proxy_tau": config.get("proxy_tau", 1.0),
            "use_cluster_pooling": config.get("use_cluster_pooling", True),
            "cluster_centroid_weight": config.get("cluster_centroid_weight", 1.0),
            "use_attn_weight_cluster_pooling": config.get("use_attn_weight_cluster_pooling", False),
            "proxy_init_std": config.get("proxy_init_std", 0.02),
        }

        # Merge with kwargs (kwargs take precedence)
        merged_params = {**proxy_params, **kwargs}

        # Create model (will load base ColBERT from path)
        model = cls(model_name_or_path=path, **merged_params)

        # Load proxy embeddings
        proxy_path = os.path.join(path, "proxy_embeddings")
        if os.path.exists(proxy_path):
            loaded_proxy = ProxyEmbeddingsModule.load(proxy_path)
            model._proxy_embeddings.weight.data.copy_(loaded_proxy.weight.data)
            logger.info(f"Loaded proxy embeddings from {proxy_path}")
        else:
            logger.warning(f"No proxy embeddings found at {proxy_path}, using initialized values")

        return model

