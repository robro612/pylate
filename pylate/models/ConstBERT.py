from __future__ import annotations

import json
import logging
import os
from typing import Iterable, Optional, override

import torch
from torch import nn
from sentence_transformers.models import Router

from ..hf_hub.model_card import PylateModelCardData
from ..scores import SimilarityFunction
from .colbert import ColBERT

logger = logging.getLogger(__name__)


class ConstBERTProjection(nn.Module):
    """
    Projects document embeddings from variable length (M, D) to fixed size (C, D).
    
    Parameters
    ----------
    max_doc_length : int
        Maximum document length to consider (M)
    output_length : int  
        Fixed output length for documents (C)
    embedding_dim : int
        Embedding dimension (D)
    bias : bool
        Whether to include bias in the projection layer. Default is False.
    """
    
    def __init__(
        self, 
        max_doc_length: int,
        output_length: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        self.max_doc_length = max_doc_length
        self.output_length = output_length
        self.embedding_dim = embedding_dim
        self.bias = bias
        
        # Projection layer: (M * D) -> (C * D)
        self.projection = nn.Linear(
            max_doc_length * embedding_dim, 
            output_length * embedding_dim,
            bias=bias
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply projection to document embeddings.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Document embeddings of shape (batch, seq_len, embedding_dim)
            
        Returns
        -------
        torch.Tensor
            Projected embeddings of shape (batch, output_length, embedding_dim)
        """
        batch_size, seq_len, emb_dim = embeddings.shape
        
        # Pad or truncate to max_doc_length
        if seq_len < self.max_doc_length:
            # Pad with zeros
            padding = torch.zeros(
                batch_size,
                self.max_doc_length - seq_len, 
                emb_dim,
                device=embeddings.device,
                dtype=embeddings.dtype
            )
            embeddings = torch.cat([embeddings, padding], dim=1)
        else:
            embeddings = embeddings[:, :self.max_doc_length, :]
        
        # Flatten: (B, M, D) -> (B, M * D)
        embeddings_flat = embeddings.reshape(batch_size, -1)
        
        # Project: (B, M * D) -> (B, C * D)
        embeddings_projected = self.projection(embeddings_flat)
        
        # Reshape: (B, C * D) -> (B, C, D)
        embeddings_out = embeddings_projected.reshape(
            batch_size, self.output_length, self.embedding_dim
        )
        
        return embeddings_out
    
    def save(self, output_path: str, safe_serialization: bool = True):
        """
        Save the ConstBERT projection module.
        
        Parameters
        ----------
        output_path : str
            Path to save the module
        safe_serialization : bool
            Whether to use safetensors format
        """
        import os
        import json
        
        os.makedirs(output_path, exist_ok=True)
        
        # Save config
        config = {
            "max_doc_length": self.max_doc_length,
            "output_length": self.output_length,
            "embedding_dim": self.embedding_dim,
            "bias": self.bias,
        }
        
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Save weights
        if safe_serialization:
            from safetensors.torch import save_file
            save_file(self.state_dict(), os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
    
    @staticmethod
    def load(input_path: str) -> "ConstBERTProjection":
        """
        Load a ConstBERT projection module.
        
        Parameters
        ----------
        input_path : str
            Path to load the module from
            
        Returns
        -------
        ConstBERTProjection
            Loaded projection module
        """
        import os
        import json
        
        # Load config
        with open(os.path.join(input_path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create module
        module = ConstBERTProjection(
            max_doc_length=config["max_doc_length"],
            output_length=config["output_length"],
            embedding_dim=config["embedding_dim"],
            bias=config.get("bias", False),
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
        
        module.load_state_dict(state_dict)
        return module


class ConstBERT(ColBERT):
    """
    ConstBERT: ColBERT with fixed-size document compression.
    
    This model extends ColBERT by projecting document embeddings from variable 
    length (M tokens) to a fixed size (C tokens) using a learned linear projection.
    Query embeddings are unchanged.
    
    Parameters
    ----------
    model_name_or_path : str | None
        Model name or path (same as ColBERT)
    constbert_output_length : int
        Fixed output length for compressed documents (C), default 64
    constbert_dim : int | None
        Embedding dimension. If None, uses the model's embedding_size
    **kwargs
        All other ColBERT parameters
    
    Examples
    --------
    >>> from pylate import models, losses
    
    >>> model = models.ConstBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    ...     constbert_output_length=64,
    ...     device="cpu",
    ... )
    
    >>> # Use with standard losses - they just need is_query parameter
    >>> loss_fn = losses.Distillation(model=model)
    """
    
    def __init__(
        self,
        model_name_or_path: str | None = None,
        constbert_output_length: int = 32,
        constbert_dim: int | None = None,
        **kwargs
    ) -> None:
        # Store ConstBERT parameters before calling super().__init__
        self.constbert_output_length = constbert_output_length
        self.constbert_dim = constbert_dim
        
        # Extract bias parameter if provided (default is False from ColBERT)
        bias = kwargs.get('bias', False)
        
        # Initialize ColBERT with all other parameters
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        
        # Determine embedding dimension
        final_embedding_dim = self.constbert_dim or self[-1].out_features
        
        # Create ConstBERT projection layer using the same bias as other Dense layers
        self.constbert_projection = ConstBERTProjection(
            max_doc_length=self.document_length,
            output_length=self.constbert_output_length,
            embedding_dim=final_embedding_dim,
            bias=bias,
        )
        
        logger.info(
            f"Initialized ConstBERT projection: "
            f"{self.document_length} -> {self.constbert_output_length} tokens"
        )
    
    @override
    def forward(self, input: dict[str, torch.Tensor], is_query: bool = False, **kwargs) -> dict[str, torch.Tensor]:
        """
        Forward pass with optional ConstBERT projection.
        
        This method:
        1. Iterates through child modules (like Transformer, Dense), skipping constbert_projection
        2. After base ColBERT processing, applies ConstBERT projection only to documents
        
        Parameters
        ----------
        input : dict[str, torch.Tensor]
            Input features from tokenizer
        is_query : bool
            If False, applies ConstBERT projection to compress documents.
            If True, passes through without projection (queries unchanged).
        **kwargs
            Additional arguments passed to child modules
            
        Returns
        -------
        dict[str, torch.Tensor]
            Output features with 'token_embeddings' key
        """
        # Extract is_query from kwargs if present (can come from either source)
        if 'is_query' in kwargs:
            is_query = kwargs.pop('is_query')
        
        # Manually iterate through child modules, skipping constbert_projection
        # (This is what SentenceTransformer.forward does, but we need to skip our projection)
        from sentence_transformers.models import Router
        
        for module_name, module in self.named_children():
            module_kwargs = {}
            if isinstance(module, Router):
                module_kwargs = kwargs
            elif module_name == 'constbert_projection':
                # Skip the projection in this loop - we'll apply it manually below
                continue
            else:
                module_kwarg_keys = []
                if self.module_kwargs is not None:
                    module_kwarg_keys = self.module_kwargs.get(module_name, [])
                module_kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key in module_kwarg_keys or (hasattr(module, "forward_kwargs") and key in module.forward_kwargs)
                }
            input = module(input, **module_kwargs)
        
        # Now apply ConstBERT projection only to documents
        if not is_query:
            embeddings = input["token_embeddings"]
            embeddings = self.constbert_projection(embeddings)
            input["token_embeddings"] = embeddings
            
            # Update attention_mask to match new length
            batch_size = embeddings.size(0)
            new_length = embeddings.size(1)
            input["attention_mask"] = torch.ones(
                batch_size, new_length,
                dtype=input["attention_mask"].dtype,
                device=input["attention_mask"].device
            )

            # Update input_ids to match new length
            input["input_ids"] = input["input_ids"][:, :new_length]
        
        return input

    @override
    def tokenize(
        self,
        texts,
        is_query: bool = True,
        pad: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not is_query:
            pad = True
        return super().tokenize(texts, is_query=is_query, pad=pad)

    def save(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ) -> None:
        """Save ConstBERT model including projection layer."""
        # Save base ColBERT components (this will automatically save constbert_projection as a child module)
        super().save(
            path=path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )
        
        # Update config to include ConstBERT parameters
        config_path = os.path.join(path, "config_sentence_transformers.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        config["model_type"] = "ConstBERT"
        config["constbert_output_length"] = self.constbert_output_length
        config["constbert_dim"] = self.constbert_dim
        # Store bias from the projection layer
        config["bias"] = self.constbert_projection.bias
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved ConstBERT model to {path}")
    
    @staticmethod
    def load(input_path: str) -> "ConstBERT":
        """Load a ConstBERT model from disk."""
        # The constbert_projection is automatically loaded as a child module
        # We just need to initialize ConstBERT with the right parameters from config
        config_path = os.path.join(input_path, "config_sentence_transformers.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create ConstBERT instance - this will load all modules including constbert_projection
        model = ConstBERT(
            model_name_or_path=input_path,
            constbert_output_length=config.get("constbert_output_length", 32),
            constbert_dim=config.get("constbert_dim", None),
            bias=config.get("bias", False),
        )
        
        logger.info(f"Loaded ConstBERT model from {input_path}")
        
        return model

