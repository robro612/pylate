from __future__ import annotations

import numpy as np
import torch

from ..utils.tensor import convert_to_tensor


def colbert_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes the ColBERT scores between queries and documents embeddings. The score is computed as the sum of maximum similarities
    between the query and the document.

    Parameters
    ----------
    queries_embeddings
        The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
    documents_embeddings
        The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)
    queries_mask
        The mask for the queries embeddings. Shape: (batch_size, num tokens queries)
    documents_mask
        The mask for the documents embeddings. Shape: (batch_size, num tokens documents)

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1.], [0.], [0.], [0.]],
    ...     [[0.], [2.], [0.], [0.]],
    ...     [[0.], [0.], [3.], [0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[10.], [0.], [1.]],
    ...     [[0.], [100.], [10.]],
    ...     [[1.], [0.], [1000.]],
    ... ])

    >>> documents_mask = torch.tensor([
    ...     [1., 1., 1.],
    ...     [1., 0., 1.],
    ...     [1., 1., 1.],
    ... ])
    >>> query_mask = torch.tensor([
    ...     [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 0., 1.]
    ... ])

    >>> scores = colbert_scores(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     queries_mask=query_mask,
    ...     documents_mask=documents_mask,
    ... )

    >>> scores
    tensor([[  10.,  10., 1000.],
            [  20.,  20., 2000.],
            [  0.,  0., 0.]])

    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    scores = torch.einsum(
        "ash,bth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)
        scores = scores * documents_mask.unsqueeze(0).unsqueeze(2)
    scores = scores.max(axis=-1).values.sum(axis=-1)
    return scores


def colbert_scores_pairwise(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Computes the ColBERT score for each query-document pair. The score is computed as the sum of maximum similarities
    between the query and the document for corresponding pairs.

    Parameters
    ----------
    queries_embeddings
        The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
    documents_embeddings
        The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1.], [0.], [0.], [0.]],
    ...     [[0.], [2.], [0.], [0.]],
    ...     [[0.], [0.], [3.], [0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[10.], [0.], [1.]],
    ...     [[0.], [100.], [1.]],
    ...     [[1.], [0.], [1000.]],
    ... ])

    >>> scores = colbert_scores_pairwise(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings
    ... )

    >>> scores
    tensor([  10.,  200., 3000.])

    """
    scores = []

    for query_embedding, document_embedding in zip(
        queries_embeddings, documents_embeddings
    ):
        query_embedding = convert_to_tensor(query_embedding)
        document_embedding = convert_to_tensor(document_embedding)

        query_document_score = torch.einsum(
            "sh,th->st",
            query_embedding,
            document_embedding,
        )

        scores.append(query_document_score.max(axis=-1).values.sum())

    return torch.stack(scores, dim=0)


def colbert_kd_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor = None,
    documents_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Computes the ColBERT scores between queries and documents embeddings. This scoring function is dedicated to the knowledge distillation pipeline.

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1.], [0.], [0.], [0.]],
    ...     [[0.], [2.], [0.], [0.]],
    ...     [[0.], [0.], [3.], [0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[[10.], [0.], [1.]], [[20.], [0.], [1.]], [[30.], [0.], [1.]]],
    ...     [[[0.], [100.], [1.]], [[0.], [200.], [1.]], [[0.], [300.], [1.]]],
    ...     [[[1.], [0.], [1000.]], [[1.], [0.], [2000.]], [[10.], [0.], [3000.]]],
    ... ])
    >>> documents_mask = torch.tensor([
    ...     [[0., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ...     [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ...     [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ... ])
    >>> query_mask = torch.tensor([
    ...     [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 0., 1.]
    ... ])
    >>> colbert_kd_scores(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     queries_mask=query_mask,
    ...     documents_mask=documents_mask,
    ... )
    tensor([[ 1.,  20.,  30.],
            [200., 400., 600.],
            [  0.,   0.,   0.]])

    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    scores = torch.einsum(
        "ash,abth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        mask = convert_to_tensor(documents_mask)
        scores = scores * mask.unsqueeze(2)

    scores = scores.max(axis=-1).values.sum(axis=-1)
    return scores


# Adapted from PrimeQA (https://github.com/primeqa/primeqa branch:xtr)
# Specifically: https://github.com/primeqa/primeqa/blob/bb9385fa129a0dbb3c7aae96ad3c782913f8280d/primeqa/ir/dense/xtr_top/xtr/modeling/XTR.py

#   Copyright 2026 IBM PrimeQA Authors
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#        http://www.apache.org/licenses/LICENSE-2.0
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Changes:
# - extricated scoring function from E2E modeling class that also handled contrastive loss computation.
# - fixed a bug in the original implementation where the alignment mask was not being applied correctly.


class XTRScores:
    """XTR scoring using global top-k token retrieval.

    Each query's top-k token matches are selected globally across all Q*N documents in the
    batch, simulating retrieval from an index. Returns the full (Q, Q*N) cross-product score
    matrix so that all in-batch documents compete as negatives.

    Parameters
    ----------
    k
        Controls top-k token matching. Accepts:
        - ``int``: single k value, returns a single score tensor.
        - ``list[int]``: multiple k values with equal weights, returns
          ``list[tuple[Tensor, float]]``.
        - ``list[tuple[int, float]]``: multiple k values with explicit weights
          (normalized to sum to 1), returns ``list[tuple[Tensor, float]]``.

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1., 0.], [0., 0.]],
    ...     [[0., 1.], [0., 0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]],
    ...     [[[0., 1.], [1., 0.]], [[1., 0.], [0., 1.]]],
    ... ])

    >>> scores = XTRScores(k=2)(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ... )
    >>> scores.shape
    torch.Size([2, 4])

    Multi-k returns a list of (scores, weight) tuples:

    >>> result = XTRScores(k=[1, 2])(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ... )
    >>> len(result)
    2
    >>> result[0][0].shape
    torch.Size([2, 4])

    """

    requires_full_batch = True

    def __init__(self, k: int | list[int] | list[tuple[int, float]] = 128):
        if isinstance(k, int):
            self._k_weights: list[tuple[int, float]] | None = None
            self.k = k
        else:
            # list[int] or list[tuple[int, float]]
            if isinstance(k[0], int):
                w = 1.0 / len(k)
                self._k_weights = [(kv, w) for kv in k]
            else:
                total = sum(w for _, w in k)
                self._k_weights = [(kv, w / total) for kv, w in k]
            self.k = max(kv for kv, _ in self._k_weights)

    def compile(self, *args, **kwargs):
        self.__call__ = torch.compile(self.__call__, *args, **kwargs)

    def _score_for_k(
        self,
        clubbed: torch.Tensor,
        k: int,
        Qb: int,
        Db: int,
        Dt: int,
        queries_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute XTR scores from pre-computed clubbed token scores for a single k."""
        _, indices = clubbed.half().topk(k, dim=-1, sorted=False)
        mask = torch.zeros_like(clubbed, dtype=torch.bool).scatter_(-1, indices, True)
        masked = clubbed * mask
        topk_scores_max = masked.view(Qb, -1, Db, Dt).max(dim=-1).values

        if queries_mask is not None:
            topk_scores_max = topk_scores_max * queries_mask.unsqueeze(-1)

        scores_sum = topk_scores_max.sum(dim=1)
        Z = topk_scores_max.gt(0).float().sum(dim=1).clamp_(min=1e-3)
        return (scores_sum / Z).float()

    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | list[tuple[torch.Tensor, float]]:
        queries_embeddings = convert_to_tensor(queries_embeddings)
        documents_embeddings = convert_to_tensor(documents_embeddings)

        Qb = queries_embeddings.shape[0]
        Dq, N = documents_embeddings.shape[:2]
        Db = Dq * N
        Qt = queries_embeddings.shape[1]
        Dt = documents_embeddings.shape[-2]
        H = queries_embeddings.shape[-1]

        docs_flat = documents_embeddings.view(Db, Dt, H)

        # Single large matmul — tensor core friendly
        Q_flat = queries_embeddings.reshape(Qb * Qt, H)
        D_flat = docs_flat.reshape(Db * Dt, H).T
        scores = (Q_flat @ D_flat).view(Qb, Qt, Db, Dt)

        if documents_mask is not None:
            docs_mask_flat = documents_mask.view(Db, Dt)
            scores = scores.masked_fill(
                ~docs_mask_flat.bool().unsqueeze(0).unsqueeze(0), -99999
            )

        clubbed = scores.flatten(2, 3)  # (Qb, Qt, Db*Dt)

        if self._k_weights is None:
            return self._score_for_k(clubbed, self.k, Qb, Db, Dt, queries_mask)

        return [
            (self._score_for_k(clubbed, k, Qb, Db, Dt, queries_mask), w)
            for k, w in self._k_weights
        ]


class XTRKDScores:
    """XTR scores for knowledge distillation. Same global top-k scoring as
    :class:`XTRScores`, but returns only each query's own N-way document scores
    to match the ``(Q, N)`` interface expected by :class:`~pylate.losses.Distillation`.

    Parameters
    ----------
    k
        Number of top token matches to retain per query token.
    """

    requires_full_batch = True

    def __init__(self, k: int = 128):
        self._xtr_scores = XTRScores(k=k)

    @property
    def k(self):
        return self._xtr_scores.k

    @k.setter
    def k(self, value):
        self._xtr_scores.k = value

    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        documents_embeddings = convert_to_tensor(documents_embeddings)
        Q, N = documents_embeddings.shape[:2]

        # Full cross-product scores: (Q, Q*N)
        all_scores = self._xtr_scores(
            queries_embeddings,
            documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
        )

        # Slice out each query's own N documents
        def _slice(scores: torch.Tensor) -> torch.Tensor:
            idx = torch.arange(Q, device=scores.device).unsqueeze(1) * N + torch.arange(
                N, device=scores.device
            )
            return scores.gather(1, idx)

        if isinstance(all_scores, list):
            return [(_slice(s), w) for s, w in all_scores]
        return _slice(all_scores)
