"""Distillation loss for ProxyAttentionColBERT model."""

from __future__ import annotations

from typing import Callable, Iterable

import torch

from ..models import ProxyAttentionColBERT
from ..scores import colbert_kd_scores


class ProxyAttentionDistillation(torch.nn.Module):
    """
    Distillation loss for ProxyAttentionColBERT model.
    
    This loss is similar to the standard Distillation loss but handles the
    fact that ProxyAttentionColBERT outputs fixed-size document embeddings
    after selection (num_select_tokens), so document masks are always all-ones.
    
    Parameters
    ----------
    model : ProxyAttentionColBERT
        The ProxyAttentionColBERT model to train
    score_metric : Callable
        Function that returns a score between two sequences of embeddings
    size_average : bool
        Average by the size of the mini-batch or perform sum
    normalize_scores : bool
        Whether to normalize scores before computing loss
        
    Examples
    --------
    >>> from pylate import models, losses
    
    >>> model = models.ProxyAttentionColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    ...     num_proxy_tokens=32,
    ...     num_select_tokens=32,
    ... )
    
    >>> loss_fn = losses.ProxyAttentionDistillation(model=model)
    """
    
    def __init__(
        self,
        model: ProxyAttentionColBERT,
        score_metric: Callable = colbert_kd_scores,
        size_average: bool = True,
        normalize_scores: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.score_metric = score_metric
        self.normalize_scores = normalize_scores
        self.loss_function = torch.nn.KLDivLoss(
            reduction="batchmean" if size_average else "sum",
            log_target=True
        )
    
    def forward(
        self,
        sentence_features: Iterable[dict[str, torch.Tensor]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the distillation loss.
        
        Parameters
        ----------
        sentence_features : Iterable[dict[str, torch.Tensor]]
            List of tokenized sentences. First is query, rest are documents.
        labels : torch.Tensor
            Target logits/scores for distillation
            
        Returns
        -------
        torch.Tensor
            The KL divergence loss between predicted and target score distributions
        """
        # Encode queries (standard encoding)
        queries_output = self.model(sentence_features[0], is_query=True)
        queries_embeddings = torch.nn.functional.normalize(
            queries_output["token_embeddings"], p=2, dim=-1
        )
        
        # Encode documents (with proxy attention selection)
        documents_output = self.model(sentence_features[1], is_query=False)
        documents_embeddings = torch.nn.functional.normalize(
            documents_output["token_embeddings"], p=2, dim=-1
        )
        
        # Reshape documents to (batch_size, n_ways, num_tokens, hidden_dim)
        documents_embeddings = documents_embeddings.view(
            queries_embeddings.size(0), -1, *documents_embeddings.shape[1:]
        )
        
        # Get skiplist and query expansion settings
        do_query_expansion = (
            self.model.do_query_expansion
            if hasattr(self.model, "do_query_expansion")
            else self.model.module.do_query_expansion
        )

        # Query mask: use attention mask (no skiplist filtering for queries)
        if not do_query_expansion:
            queries_mask = sentence_features[0]["attention_mask"].bool()
        else:
            queries_mask = None

        # Document mask: for ProxyAttentionColBERT, after selection all positions are valid
        # Use the output attention_mask from the model (all ones for selected tokens)
        doc_mask = documents_output.get("attention_mask", None)
        if doc_mask is not None:
            documents_mask = doc_mask.view(
                queries_embeddings.size(0), -1, *doc_mask.shape[1:]
            )
        else:
            # All positions valid (fallback)
            documents_mask = None
        
        # Compute ColBERT scores
        scores = self.score_metric(
            queries_embeddings,
            documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
        )
        
        # Optionally normalize scores
        if self.normalize_scores:
            max_scores, _ = torch.max(scores, dim=1, keepdim=True)
            min_scores, _ = torch.min(scores, dim=1, keepdim=True)
            epsilon = 1e-8
            scores = (scores - min_scores) / (max_scores - min_scores + epsilon)
        
        # KL divergence loss
        return self.loss_function(
            torch.nn.functional.log_softmax(scores, dim=-1),
            torch.nn.functional.log_softmax(labels, dim=-1),
        )

