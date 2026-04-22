from __future__ import annotations

from typing import Callable, Iterable, Literal

import torch

from ..models import ColBERT
from ..scores import ScopedBatchScores, colbert_kd_scores
from .contrastive import extract_skiplist_mask


class Distillation(torch.nn.Module):
    """Distillation loss for ColBERT model. The loss is computed with respect to the format of SentenceTransformer library.

    Parameters
    ----------
    model
        SentenceTransformer model.
    score_metric
        Function that returns a score between two sequences of embeddings.
    size_average
        Average by the size of the mini-batch or perform sum.
    normalize_scores
        Min-max normalization mode before computing the KL loss. Supported values:
        - ``False``: normalize neither student nor teacher scores.
        - ``"student"``: normalize only student scores.
        - ``"teacher"``: normalize only teacher scores.
        - ``True`` or ``"both"``: normalize both student and teacher scores.
    temperature
        Temperature to divide scores by before log_softmax.

    Examples
    --------
    >>> from pylate import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> distillation = losses.Distillation(model=model)

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

    >>> loss = distillation(sentence_features=sentence_features, labels=labels)

    >>> assert isinstance(loss.item(), float)
    """

    def __init__(
        self,
        model: ColBERT,
        score_metric: Callable = colbert_kd_scores,
        size_average: bool = True,
        normalize_scores: bool | Literal["student", "teacher", "both"] = True,
        temperature: float = 1.0,
    ) -> None:
        super(Distillation, self).__init__()
        self.score_metric = score_metric
        self.model = model
        self.loss_function = torch.nn.KLDivLoss(
            reduction="batchmean" if size_average else "sum", log_target=True
        )
        if isinstance(normalize_scores, bool):
            # Backward-compatible behavior: True means normalize both.
            self.normalize_scores = "both" if normalize_scores else "none"
        else:
            mode = normalize_scores.lower()
            allowed_modes = {"student", "teacher", "both"}
            if mode not in allowed_modes:
                raise ValueError(
                    "normalize_scores must be one of False, 'student', 'teacher', or 'both'."
                )
            self.normalize_scores = mode
        self.temperature = temperature

    @staticmethod
    def _minmax_normalize(scores: torch.Tensor) -> torch.Tensor:
        max_scores, _ = torch.max(scores, dim=1, keepdim=True)
        min_scores, _ = torch.min(scores, dim=1, keepdim=True)
        epsilon = 1e-8
        return (scores - min_scores) / (max_scores - min_scores + epsilon)

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor
    ) -> torch.Tensor:
        """Computes the distillation loss with respect to SentenceTransformer.

        Parameters
        ----------
        sentence_features
            List of tokenized sentences. The first sentence is the query and the rest are documents.
        labels
            The logits for the distillation loss.

        """
        queries_embeddings = torch.nn.functional.normalize(
            self.model(sentence_features[0])["token_embeddings"], p=2, dim=-1
        )
        # Compute the bs * n_ways embeddings
        documents_embeddings = torch.nn.functional.normalize(
            self.model(sentence_features[1])["token_embeddings"], p=2, dim=-1
        )

        # Reshape them to (bs, n_ways)
        documents_embeddings = documents_embeddings.view(
            queries_embeddings.size(0), -1, *documents_embeddings.shape[1:]
        )

        # handle the model being wrapped in (D)DP and so require to access module first
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
        if isinstance(scores, list):
            scores = sum(w * s for s, w in scores)
        teacher_scores = labels.float()
        if self.normalize_scores in {"student", "both"}:
            scores = self._minmax_normalize(scores)
        if self.normalize_scores in {"teacher", "both"}:
            teacher_scores = self._minmax_normalize(teacher_scores)
        return self.loss_function(
            torch.nn.functional.log_softmax(scores / self.temperature, dim=-1),
            torch.nn.functional.log_softmax(teacher_scores, dim=-1),
        )


class Distillation_New(Distillation):
    """ScopedBatchScores-integrated variant of Distillation."""

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor
    ) -> torch.Tensor:
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
        if not isinstance(self.score_metric, ScopedBatchScores):
            raise TypeError("Distillation_New requires score_metric=ScopedBatchScores.")
        scores = self.score_metric(
            queries_embeddings,
            documents_embeddings,
            queries_mask=masks[0] if not do_query_expansion else None,
            documents_mask=documents_embeddings_mask,
            scoring_scope="global",
            return_scope="local",
        )
        if isinstance(scores, list):
            raise TypeError(
                "Distillation_New expects a single score tensor, not weighted multi-k scores."
            )
        teacher_scores = labels.float()
        if self.normalize_scores in {"student", "both"}:
            scores = self._minmax_normalize(scores)
        if self.normalize_scores in {"teacher", "both"}:
            teacher_scores = self._minmax_normalize(teacher_scores)
        return self.loss_function(
            torch.nn.functional.log_softmax(scores / self.temperature, dim=-1),
            torch.nn.functional.log_softmax(teacher_scores, dim=-1),
        )
