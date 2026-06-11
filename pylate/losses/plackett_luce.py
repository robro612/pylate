from __future__ import annotations

from typing import Callable, Iterable

import torch

from ..models import ColBERT
from ..scores import colbert_kd_scores
from .contrastive import extract_skiplist_mask


class PlackettLuce(torch.nn.Module):
    """Plackett-Luce distillation loss for ColBERT models.

    Implements the list-wise ranking loss from PLD (arXiv:2506.12542). The teacher
    scores define a permutation σ with the positive document (index 0) placed first,
    followed by the remaining documents sorted by descending teacher score. The loss
    is the weighted negative PL log-likelihood under that permutation, where each step
    is weighted by the teacher's softmax confidence.

    This loss is translation-invariant by construction, so score normalization is not
    required (unlike the KL-divergence-based Distillation loss).

    Parameters
    ----------
    model
        ColBERT model being trained.
    score_metric
        Function that returns scores between query and document embeddings.
    size_average
        Average loss over the batch (True) or sum (False).

    Examples
    --------
    >>> from pylate import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> loss_fn = losses.PlackettLuce(model=model)

    >>> query = model.tokenize([
    ...     "fruits are healthy.",
    ... ], is_query=True)

    >>> documents = model.tokenize([
    ...     "fruits are good for health.",
    ...     "fruits are bad for health."
    ... ], is_query=False)

    >>> sentence_features = [query, documents]

    >>> labels = torch.tensor([
    ...     [0.7, 0.3],
    ... ], dtype=torch.float32)

    >>> loss = loss_fn(sentence_features=sentence_features, labels=labels)

    >>> assert isinstance(loss.item(), float)
    """

    def __init__(
        self,
        model: ColBERT,
        score_metric: Callable = colbert_kd_scores,
        size_average: bool = True,
    ) -> None:
        super(PlackettLuce, self).__init__()
        self.score_metric = score_metric
        self.model = model
        self.size_average = size_average

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor
    ) -> torch.Tensor:
        """Computes the Plackett-Luce distillation loss.

        Parameters
        ----------
        sentence_features
            List of tokenized sentences. The first element is the query; the second
            contains all documents (positive first, then negatives).
        labels
            Teacher scores, shape (batch_size, num_docs). The document at index 0
            in each row is treated as the positive and placed first in the PL ranking.
        """
        queries_embeddings = torch.nn.functional.normalize(
            self.model(sentence_features[0])["token_embeddings"], p=2, dim=-1
        )
        documents_embeddings = torch.nn.functional.normalize(
            self.model(sentence_features[1])["token_embeddings"], p=2, dim=-1
        )
        documents_embeddings = documents_embeddings.view(
            queries_embeddings.size(0), -1, *documents_embeddings.shape[1:]
        )

        skiplist = (
            self.model.skiplist
            if hasattr(self.model, "skiplist")
            else self.model.module.skiplist
        )
        do_query_expansion = (
            self.model.do_query_expansion
            if hasattr(self.model, "do_query_expansion")
            else self.model.module.do_query_expansion
        )

        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=skiplist
        )
        documents_embeddings_mask = masks[1].view(
            queries_embeddings.size(0), -1, *masks[1].shape[1:]
        )
        scores = self.score_metric(
            queries_embeddings,
            documents_embeddings,
            queries_mask=masks[0] if not do_query_expansion else None,
            documents_mask=documents_embeddings_mask,
        )

        # Build permutation σ: positive (index 0) first, then remaining docs sorted
        # by descending teacher score — matching the PLD paper's ranking construction.
        B, N = labels.shape
        remaining_order = labels[:, 1:].argsort(dim=-1, descending=True) + 1  # (B, N-1)
        positive_idx = torch.zeros(B, 1, dtype=torch.long, device=labels.device)
        order = torch.cat([positive_idx, remaining_order], dim=-1)  # (B, N)

        sorted_scores = scores.gather(1, order)  # student scores in PL order

        # log P(step i) = sorted_scores[i] - logsumexp(sorted_scores[i:])
        # Computed efficiently via logcumsumexp on the reversed sequence.
        suffix_lse = torch.logcumsumexp(sorted_scores.flip(-1), dim=-1).flip(-1)
        pl_log_probs = sorted_scores - suffix_lse  # (B, N)

        # Weight each step by teacher softmax confidence at that ranked position.
        weights = labels.softmax(dim=-1).gather(1, order)  # (B, N)

        per_example_loss = -(weights * pl_log_probs).sum(dim=-1)  # (B,)
        return per_example_loss.mean() if self.size_average else per_example_loss.sum()
