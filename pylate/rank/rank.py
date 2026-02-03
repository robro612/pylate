from __future__ import annotations

import numpy as np
import torch
from typing_extensions import TypedDict

from ..scores import colbert_scores
from ..utils import convert_to_tensor as func_convert_to_tensor


class RerankResult(TypedDict):
    """
    Rerank result for ranking.

    Parameters
    ----------
    id
        The document id.
    score
        The document score.
    """

    id: int | str
    score: float


def reshape_embeddings(
    embeddings: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Reshape the embeddings to the correct shape."""
    if isinstance(embeddings, torch.Tensor):
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(dim=0)

    elif isinstance(embeddings, np.ndarray):
        if len(embeddings.shape) == 2:
            return np.expand_dims(a=embeddings, axis=0)

    return embeddings


def rerank(
    documents_ids: list[list[int | str]],
    queries_embeddings: list[list[float | int] | np.ndarray | torch.Tensor],
    documents_embeddings: list[list[float | int] | np.ndarray | torch.Tensor],
    device: str = None,
) -> list[list[RerankResult]]:
    """Rerank the documents based on the queries embeddings.

    Parameters
    ----------
    documents_ids
        The documents ids.
    queries_embeddings
        The queries embeddings which is a dictionary of queries and their embeddings.
    documents_embeddings
        The documents embeddings which is a dictionary of documents ids and their embeddings.
    device
        The device to use for the reranking. If None, the device of the queries embeddings will be used.

    Examples
    --------
    >>> from pylate import models, rank

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> queries = [
    ...     "query A",
    ...     "query B",
    ... ]

    >>> documents = [
    ...     ["document A", "document B"],
    ...     ["document 1", "document C", "document B"],
    ... ]

    >>> documents_ids = [
    ...    [1, 2],
    ...    [1, 3, 2],
    ... ]

    >>> queries_embeddings = model.encode(
    ...     queries,
    ...     is_query=True,
    ...     batch_size=1,
    ... )

    >>> documents_embeddings = model.encode(
    ...     documents,
    ...     is_query=False,
    ...     batch_size=1,
    ... )

    >>> reranked_documents = rank.rerank(
    ...     documents_ids=documents_ids,
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> assert isinstance(reranked_documents, list)
    >>> assert len(reranked_documents) == 2
    >>> assert len(reranked_documents[0]) == 2
    >>> assert len(reranked_documents[1]) == 3
    >>> assert isinstance(reranked_documents[0], list)
    >>> assert isinstance(reranked_documents[0][0], dict)
    >>> assert "id" in reranked_documents[0][0]
    >>> assert "score" in reranked_documents[0][0]

    """
    results = []

    queries_embeddings = reshape_embeddings(embeddings=queries_embeddings)
    documents_embeddings = reshape_embeddings(embeddings=documents_embeddings)

    for query_embeddings, query_documents_ids, query_documents_embeddings in zip(
        queries_embeddings, documents_ids, documents_embeddings
    ):
        query_embeddings = func_convert_to_tensor(query_embeddings)

        query_documents_embeddings = [
            func_convert_to_tensor(query_document_embeddings)
            for query_document_embeddings in query_documents_embeddings
        ]

        # Pad the documents embeddings
        query_documents_embeddings = torch.nn.utils.rnn.pad_sequence(
            query_documents_embeddings, batch_first=True, padding_value=0
        )

        if device is not None:
            query_embeddings = query_embeddings.to(device)
            query_documents_embeddings = query_documents_embeddings.to(device)
        else:
            query_documents_embeddings = query_documents_embeddings.to(
                query_embeddings.device
            )

        query_scores = colbert_scores(
            queries_embeddings=query_embeddings.unsqueeze(0),
            documents_embeddings=query_documents_embeddings,
        )[0]

        scores, sorted_indices = torch.sort(input=query_scores, descending=True)
        scores = scores.cpu().tolist()

        query_documents = [query_documents_ids[idx] for idx in sorted_indices.tolist()]

        results.append(
            [
                RerankResult(id=doc_id, score=score)
                for doc_id, score in zip(query_documents, scores)
            ]
        )

    return results

def _compute_imputation_scores(
    query_scores: list[list[float]],
    imputation: str,
    percentile: float,
    power_law_multiplier: float,
    device: str,
) -> torch.Tensor:
    """Compute imputation scores for each query token.

    Parameters
    ----------
    query_scores
        List of length q_tok, where each element is a list of scores.
    imputation
        Imputation strategy: "min", "zero", "mean", "percentile", or "power_law".
    percentile
        Percentile value (0-100) for percentile imputation.
    power_law_multiplier
        Multiplier for k' when extrapolating power-law (e.g., 100 means extrapolate to rank 100*k').
    device
        Device for tensor computation.

    Returns
    -------
    torch.Tensor
        Imputation score for each query token, shape (q_tok,).
    """
    q_tok = len(query_scores)

    if imputation == "zero":
        return torch.zeros(q_tok, dtype=torch.float32, device=device)

    elif imputation == "min":
        return torch.tensor(
            [min(scores) if len(scores) > 0 else 0.0 for scores in query_scores],
            dtype=torch.float32,
            device=device,
        )

    elif imputation == "mean":
        return torch.tensor(
            [sum(scores) / len(scores) if len(scores) > 0 else 0.0 for scores in query_scores],
            dtype=torch.float32,
            device=device,
        )

    elif imputation == "percentile":
        imputation_scores = []
        for scores in query_scores:
            if len(scores) == 0:
                imputation_scores.append(0.0)
            else:
                # np.percentile expects percentile in [0, 100]
                imputation_scores.append(float(np.percentile(scores, percentile)))
        return torch.tensor(imputation_scores, dtype=torch.float32, device=device)

    elif imputation == "power_law":
        # Fit power-law: score(rank) = a * rank^(-b)
        # In log space: log(score) = log(a) - b * log(rank)
        # Extrapolate to rank 100 * k' as per Lee et al., 2023
        imputation_scores = []
        for scores in query_scores:
            if len(scores) < 2:
                # Not enough points to fit, fall back to min
                imputation_scores.append(min(scores) if len(scores) > 0 else 0.0)
                continue

            # Sort scores descending (rank 1 = highest score)
            sorted_scores = sorted(scores, reverse=True)
            k_prime = len(sorted_scores)

            # Filter out non-positive scores (can't take log)
            valid_pairs = [
                (rank, score)
                for rank, score in enumerate(sorted_scores, start=1)
                if score > 0
            ]

            if len(valid_pairs) < 2:
                imputation_scores.append(min(scores) if len(scores) > 0 else 0.0)
                continue

            ranks, valid_scores = zip(*valid_pairs)
            log_ranks = np.log(ranks)
            log_scores = np.log(valid_scores)

            # Linear regression in log-log space: log(score) = log(a) - b * log(rank)
            # Using numpy's polyfit for degree 1 polynomial
            try:
                coeffs = np.polyfit(log_ranks, log_scores, 1)
                neg_b, log_a = coeffs  # slope is -b, intercept is log(a)

                # Extrapolate to rank power_law_multiplier * k'
                extrapolate_rank = power_law_multiplier * k_prime
                log_imputed = log_a + neg_b * np.log(extrapolate_rank)
                imputed = np.exp(log_imputed)

                # Clamp to reasonable range [0, min_retrieved_score]
                imputed = max(0.0, min(float(imputed), min(scores)))
                imputation_scores.append(imputed)
            except (np.linalg.LinAlgError, ValueError):
                # Fitting failed, fall back to min
                imputation_scores.append(min(scores))

        return torch.tensor(imputation_scores, dtype=torch.float32, device=device)

    else:
        raise ValueError(
            f"Unknown imputation strategy: {imputation}. "
            f"Expected one of: 'min', 'zero', 'mean', 'percentile', 'power_law'."
        )


def score_xtr(
    query_doc_ids: list[list[str | int]],
    query_scores: list[list[float]],
    k: int,
    device: str = "cpu",
    imputation: str = "min",
    percentile: float = 10.0,
    power_law_multiplier: float = 100.0,
) -> list[RerankResult]:
    """Score documents using XTR (eXact Token Retrieval) scoring.

    XTR scoring differs from ColBERT in that it doesn't do full reranking.
    Instead, it only scores documents using initially retrieved tokens, and
    imputes missing token scores based on the chosen imputation strategy.

    Parameters
    ----------
    query_doc_ids
        List of length q_tok, where each element is a list of k_token document IDs
        retrieved for that query token. Document IDs can be strings or integers.
    query_scores
        List of length q_tok, where each element is a list of k_token scores
        corresponding to the retrieved document IDs.
    k
        Number of top documents to return.
    device
        Device to use for computation ('cpu', 'cuda', etc.).
    imputation
        Strategy for imputing missing scores. Options:
        - "min": Use minimum retrieved score per query token (default, original XTR).
        - "zero": Impute with zero (missing tokens contribute nothing).
        - "mean": Use mean of retrieved scores per query token.
        - "percentile": Use specified percentile of retrieved scores.
        - "power_law": Fit power-law curve to retrieved scores and extrapolate
          to rank (power_law_multiplier * k') as per Lee et al., 2023.
    percentile
        Percentile value (0-100) for percentile imputation. Default is 10.0.
    power_law_multiplier
        Multiplier for k' when extrapolating power-law. Default is 100.0 (extrapolate
        to rank 100*k' as in the original XTR paper).

    Returns
    -------
    list[RerankResult]
        Top-k documents sorted by score (descending).

    Notes
    -----
    The XTR scoring algorithm:
    1. For each document, sum scores across all query tokens
    2. If a document's token wasn't retrieved for a query token, use the
       imputed score based on the chosen strategy
    3. If multiple tokens from the same document were retrieved for a query token,
       use the maximum score

    Examples
    --------
    >>> from pylate.rank import score_xtr
    >>> query_doc_ids = [
    ...     ["doc1", "doc2", "doc3"],  # Retrieved for query token 0
    ...     ["doc2", "doc3", "doc4"],  # Retrieved for query token 1
    ... ]
    >>> query_scores = [
    ...     [0.9, 0.7, 0.5],  # Scores for query token 0
    ...     [0.8, 0.6, 0.4],  # Scores for query token 1
    ... ]
    >>> results = score_xtr(query_doc_ids, query_scores, k=3)
    >>> assert len(results) == 3
    >>> assert results[0]["id"] == "doc2"  # Has high scores for both tokens

    >>> # Using zero imputation
    >>> results_zero = score_xtr(query_doc_ids, query_scores, k=3, imputation="zero")

    >>> # Using power-law imputation
    >>> results_pl = score_xtr(query_doc_ids, query_scores, k=3, imputation="power_law")

    """
    q_tok = len(query_doc_ids)
    
    if q_tok == 0:
        return []
    
    # Flatten all doc IDs and scores with their query token indices
    all_doc_ids = []
    all_scores = []
    q_tok_indices = []
    
    for q_idx, (token_docs, token_scores) in enumerate(zip(query_doc_ids, query_scores)):
        all_doc_ids.extend(token_docs)
        all_scores.extend(token_scores)
        q_tok_indices.extend([q_idx] * len(token_docs))
    
    # Convert to tensors
    # Handle both string and integer document IDs
    if len(all_doc_ids) == 0:
        return []
    
    # Keep track of original doc IDs for final output
    doc_id_is_string = isinstance(all_doc_ids[0], str)
    if doc_id_is_string:
        # Create a mapping from string IDs to integers
        unique_doc_id_strings = list(set(all_doc_ids))
        doc_id_to_int = {doc_id: idx for idx, doc_id in enumerate(unique_doc_id_strings)}
        all_doc_ids_int = [doc_id_to_int[doc_id] for doc_id in all_doc_ids]
        all_doc_ids_t = torch.tensor(all_doc_ids_int, dtype=torch.long, device=device)
    else:
        all_doc_ids_t = torch.tensor(all_doc_ids, dtype=torch.long, device=device)
    
    all_scores_t = torch.tensor(all_scores, dtype=torch.float32, device=device)
    q_tok_indices_t = torch.tensor(q_tok_indices, dtype=torch.long, device=device)
    
    # Get unique document IDs
    unique_doc_ids, inverse_indices = torch.unique(all_doc_ids_t, return_inverse=True)
    num_docs = len(unique_doc_ids)
    
    # Compute imputation scores based on chosen strategy
    imputation_scores = _compute_imputation_scores(
        query_scores=query_scores,
        imputation=imputation,
        percentile=percentile,
        power_law_multiplier=power_law_multiplier,
        device=device,
    )  # Shape: (q_tok,)

    # Step 1: Compute max actual score per (doc, query_token) pair
    # Initialize with -inf so we can detect which pairs have no retrieved score
    NEG_INF = float("-inf")
    doc_scores = torch.full(
        (num_docs, q_tok), NEG_INF, dtype=torch.float32, device=device
    )

    # Flatten for 1D scatter, then reshape
    doc_scores_flat = doc_scores.reshape(-1)
    flat_indices = inverse_indices * q_tok + q_tok_indices_t

    # Use scatter_reduce with reduce='amax' to keep max score when multiple tokens
    # from the same document are retrieved for a single query token
    doc_scores_flat.scatter_reduce_(
        0, flat_indices, all_scores_t, reduce="amax", include_self=False
    )
    doc_scores = doc_scores_flat.reshape(num_docs, q_tok)

    # Step 2: Replace -inf (no retrieved score) with imputation scores
    missing_mask = doc_scores == NEG_INF
    doc_scores = torch.where(
        missing_mask,
        imputation_scores.unsqueeze(0).expand(num_docs, q_tok),
        doc_scores,
    )
    
    # Sum across query tokens to get final document scores
    final_scores = doc_scores.sum(dim=1)
    
    # Get top k documents
    top_k_scores, top_k_indices = torch.topk(
        final_scores, k=min(k, num_docs), largest=True
    )
    top_k_doc_ids = unique_doc_ids[top_k_indices]
    
    # Convert back to original document ID format
    results = []
    for doc_id_tensor, score in zip(top_k_doc_ids, top_k_scores):
        if doc_id_is_string:
            doc_id = unique_doc_id_strings[doc_id_tensor.item()]
        else:
            doc_id = doc_id_tensor.item()
        
        results.append(
            RerankResult(id=doc_id, score=score.item())
        )
    
    return results
