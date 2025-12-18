from __future__ import annotations

import numpy as np
import torch
import wandb
from transformers import TrainerCallback

from ..utils.tensor import convert_to_tensor

def is_main_process():
    return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0

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

    Returns
    -------
    scores
        The scores between the queries and documents. Shape: (batch_size, batch_size)
    
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
    # (batch_size, batch_size, q_seq_len, d_seq_len) -> (batch_size, batch_size)
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
    # (batch_size, q_seq_len, embedding_size)
    queries_embeddings = convert_to_tensor(queries_embeddings)
    # (batch_size, n_ways, d_seq_len, embedding_size)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    # (batch_size, n_ways, batch_size, q_seq_len, d_seq_len)
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

def xtr_contrastive_training_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    k_prime: int = 100,
    use_normalizer_Z : bool = False,
    impute_scores_instead_of_zero: bool = False,
    Z_clamp_value: float = 1.0,
) -> torch.Tensor:

    # 1. Compute raw Cross-Batch Scores
    # (batch_size, batch_size, q_seq_len, d_seq_len)
    cross_batch_scores = torch.einsum(
        "aqh, bdh->abqd",
        queries_embeddings,
        documents_embeddings,
    )

    # 2. Apply Padding Masking
    if queries_mask is not None:
        cross_batch_scores = cross_batch_scores * queries_mask.unsqueeze(1).unsqueeze(3)
    
    if documents_mask is not None:
        cross_batch_scores = cross_batch_scores * documents_mask.unsqueeze(0).unsqueeze(2)

    batch_size_q, batch_size_d, q_seq_len, d_seq_len = cross_batch_scores.shape

    # 3. Flatten to find Global Top-K Thresholds
    # (batch_size, q_seq_len, batch_size * d_seq_len)
    cross_batch_scores_flattened = cross_batch_scores.permute(0, 2, 1, 3).reshape(
        batch_size_q, q_seq_len, batch_size_d * d_seq_len
    )

    # Get the threshold value for the top k_prime tokens
    k_index = max(1, cross_batch_scores_flattened.size(-1) - k_prime + 1)
    
    # (batch_size, q_seq_len) -> (batch_size, 1, q_seq_len, 1)
    thresholds = cross_batch_scores_flattened.kthvalue(k=k_index, dim=-1).values
    thresholds = thresholds.unsqueeze(1).unsqueeze(-1)

    # 4. Determine Retrieval (Alignment Matrix A)
    # A_ij = 1 if score >= threshold, else 0
    is_retrieved = cross_batch_scores >= thresholds

    retrieved_counts = is_retrieved.sum(dim=-1).float()  # shape (bq, bd, q_len)

    # 5. Compute Max Similarity for Retrieved Tokens
    # We use -inf for non-retrieved tokens so they don't affect the max.
    # (batch_size, batch_size, q_seq_len, d_seq_len)
    if impute_scores_instead_of_zero:
        masked_scores = torch.where(is_retrieved, cross_batch_scores, thresholds)
    else:
        masked_scores = cross_batch_scores.masked_fill(~is_retrieved, -float('inf'))

    # Take max over document tokens (dim -1)
    # (batch_size, batch_size, q_seq_len)
    max_sim_per_query_token = masked_scores.max(dim=-1).values

    # 6. Handle "Nothing Retrieved" Cases
    # If a query token retrieved NOTHING (all were -inf), the max is -inf.
    # We must zero these out so they don't corrupt the sum.
    # Check if at least one doc token was retrieved for this query token
    valid_retrieval_mask = is_retrieved.any(dim=-1)

    # Set score to 0.0 where nothing was retrieved
    max_sim_per_query_token = torch.where(
        valid_retrieval_mask,
        max_sim_per_query_token,
        torch.zeros_like(max_sim_per_query_token)
    )

    # 7. Compute Normalizer Z and Final Scores
    # Sum over query tokens to get the numerator
    # (batch_size, batch_size)
    numerator = max_sim_per_query_token.sum(dim=-1)

    if use_normalizer_Z:
        # Z = number of query tokens that retrieved at least one document token
        # (batch_size, batch_size)
        normalizer_Z = valid_retrieval_mask.sum(dim=-1).to(numerator.dtype)
        scores = numerator / normalizer_Z.clamp(min=Z_clamp_value)
        xtr_scores = torch.where(normalizer_Z > 0, scores, torch.zeros_like(scores, dtype=scores.dtype))
    else:
        xtr_scores = numerator

    # log the stats for debugging / monitoring
    # Create masks for positives (diagonals) and negatives (off-diagonals)
    batch_size = numerator.shape[0]
    diag_mask = torch.eye(batch_size, device=numerator.device, dtype=torch.bool)
    off_diag_mask = ~diag_mask
    
    # Aggregate retrieved_counts over q_seq_len dimension for logging
    # (batch_size, batch_size, q_seq_len) -> (batch_size, batch_size)
    retrieved_counts_agg = retrieved_counts.mean(dim=-1)
    
    # Compute statistics for all pairs (existing behavior)
    log_dict = {
        "numerator_mean": numerator.mean().item(),
        "numerator_std": numerator.std().item(),
        "numerator_min": numerator.min().item(),
        "numerator_max": numerator.max().item(),
        "xtr_scores_mean": xtr_scores.mean().item(),
        "xtr_scores_std": xtr_scores.std().item(),
        "k_prime": k_prime,
        "retrieved_counts_mean": retrieved_counts.mean().item(),
        "retrieved_counts_std": retrieved_counts.std().item(),
        "retrieved_counts_max": retrieved_counts.max().item(),
        "numerator_mean_pos": numerator[diag_mask].mean().item(),
        "numerator_std_pos": numerator[diag_mask].std().item(),
        "numerator_min_pos": numerator[diag_mask].min().item(),
        "numerator_max_pos": numerator[diag_mask].max().item(),
        "xtr_scores_mean_pos": xtr_scores[diag_mask].mean().item(),
        "xtr_scores_std_pos": xtr_scores[diag_mask].std().item(),
        "retrieved_counts_mean_pos": retrieved_counts_agg[diag_mask].mean().item(),
        "retrieved_counts_std_pos": retrieved_counts_agg[diag_mask].std().item(),
        "retrieved_counts_max_pos": retrieved_counts_agg[diag_mask].max().item(),
        "numerator_mean_neg": numerator[off_diag_mask].mean().item(),
        "numerator_std_neg": numerator[off_diag_mask].std().item(),
        "numerator_min_neg": numerator[off_diag_mask].min().item(),
        "numerator_max_neg": numerator[off_diag_mask].max().item(),
        "xtr_scores_mean_neg": xtr_scores[off_diag_mask].mean().item(),
        "xtr_scores_std_neg": xtr_scores[off_diag_mask].std().item(),
        "retrieved_counts_mean_neg": retrieved_counts_agg[off_diag_mask].mean().item(),
        "retrieved_counts_std_neg": retrieved_counts_agg[off_diag_mask].std().item(),
        "retrieved_counts_max_neg": retrieved_counts_agg[off_diag_mask].max().item(),
    }
    
    # Add Z statistics (all pairs, positives, and negatives) if normalizer is used
    if use_normalizer_Z:
        log_dict.update({
            "Z_mean": normalizer_Z.float().mean().item(),
            "Z_std": normalizer_Z.float().std().item(),
            "Z_min": normalizer_Z.min().item(),
            "Z_max": normalizer_Z.max().item(),
            "Z_mean_pos": normalizer_Z[diag_mask].float().mean().item(),
            "Z_std_pos": normalizer_Z[diag_mask].float().std().item(),
            "Z_min_pos": normalizer_Z[diag_mask].min().item(),
            "Z_max_pos": normalizer_Z[diag_mask].max().item(),
            "Z_mean_neg": normalizer_Z[off_diag_mask].float().mean().item(),
            "Z_std_neg": normalizer_Z[off_diag_mask].float().std().item(),
            "Z_min_neg": normalizer_Z[off_diag_mask].min().item(),
            "Z_max_neg": normalizer_Z[off_diag_mask].max().item(),
        })

    # log whether Z changes the max document
    # xtr_scores is (bq, bd)
    top_idx = xtr_scores.argmax(dim=1)                # which doc is top for each query
    pos_idx = torch.arange(xtr_scores.size(0), device=xtr_scores.device)
    frac_pos_top = (top_idx == pos_idx).float().mean().item()
    # how many queries have same argmax?
    same_argmax_frac = (numerator.argmax(dim=1) == xtr_scores.argmax(dim=1)).float().mean().item()

    log_dict.update({
        "frac_pos_top": frac_pos_top,
        "same_argmax_frac": same_argmax_frac,
        "k_index": k_index,
    })

    if is_main_process():
        wandb.log(log_dict)

    return xtr_scores

def xtr_kd_training_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    k_prime: int = 100,
    use_normalizer_Z: bool = False,
    Z_clamp_value: float = 1.0,
) -> torch.Tensor:
    """Computes the XTR scores for Knowledge Distillation with WandB logging.
    
    The logic mirrors the contrastive training scores:
    1. Computes cross-batch similarity to establish global thresholds.
    2. Determines which tokens are 'retrieved' based on k_prime.
    3. Returns scores only for the specific (aligned) n_ways documents provided for each query.
    4. Logs statistics separating Positives (index 0) from Negatives (indices 1..N).

    Parameters
    ----------
    queries_embeddings
        Shape: (batch_size, q_seq_len, embedding_size)
    documents_embeddings
        Shape: (batch_size, n_ways, d_seq_len, embedding_size)
    queries_mask
        Shape: (batch_size, q_seq_len)
    documents_mask
        Shape: (batch_size, n_ways, d_seq_len)
    k_prime
        The number of top tokens to consider for each query token.

    Returns
    -------
    scores
        Shape: (batch_size, n_ways)
    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    batch_size, q_seq_len, _ = queries_embeddings.shape
    _, n_ways, d_seq_len, _ = documents_embeddings.shape

    # 1. Flatten the n_ways dimension into the batch dimension for the documents
    # to perform global retrieval thresholding.
    # (batch_size * n_ways, d_seq_len, embedding_size)
    flat_documents_embeddings = documents_embeddings.reshape(-1, d_seq_len, documents_embeddings.size(-1))

    # 2. Compute raw Cross-Batch Scores
    # We compare every query against EVERY document in the batch (including negatives/n_ways)
    # to find the correct global threshold.
    # (batch_size, batch_size * n_ways, q_seq_len, d_seq_len)
    cross_batch_scores = torch.einsum(
        "aqh, bdh->abqd",
        queries_embeddings,
        flat_documents_embeddings,
    )

    # 3. Apply Padding Masking
    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        # (batch_size, 1, q_seq_len, 1)
        cross_batch_scores = cross_batch_scores * queries_mask.unsqueeze(1).unsqueeze(3)
    
    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)
        # Flatten mask: (batch_size * n_ways, d_seq_len)
        flat_documents_mask = documents_mask.reshape(-1, d_seq_len)
        # (1, batch_size * n_ways, 1, d_seq_len)
        cross_batch_scores = cross_batch_scores * flat_documents_mask.unsqueeze(0).unsqueeze(2)

    total_docs = batch_size * n_ways

    # 4. Flatten to find Global Top-K Thresholds
    # (batch_size, q_seq_len, total_docs * d_seq_len)
    cross_batch_scores_flattened = cross_batch_scores.permute(0, 2, 1, 3).reshape(
        batch_size, q_seq_len, total_docs * d_seq_len
    )

    # Get the threshold value for the top k_prime tokens
    k_index = max(1, cross_batch_scores_flattened.size(-1) - k_prime + 1)
    
    # (batch_size, q_seq_len) -> (batch_size, 1, q_seq_len, 1)
    thresholds = cross_batch_scores_flattened.kthvalue(k=k_index, dim=-1).values
    thresholds = thresholds.unsqueeze(1).unsqueeze(-1)

    # 5. Determine Retrieval (Alignment Matrix A)
    # (batch_size, total_docs, q_seq_len, d_seq_len)
    is_retrieved = cross_batch_scores >= thresholds

    # -------------------------------------------------------------------------
    # KD SPECIFIC EXTRACTION
    # -------------------------------------------------------------------------

    # Reshape back to separate batch and n_ways
    # (batch_size, batch_size, n_ways, q_seq_len, d_seq_len)
    cross_batch_scores = cross_batch_scores.view(batch_size, batch_size, n_ways, q_seq_len, d_seq_len)
    is_retrieved = is_retrieved.view(batch_size, batch_size, n_ways, q_seq_len, d_seq_len)

    # Select only the aligned pairs: Query[i] vs Docs[i]
    batch_indices = torch.arange(batch_size, device=cross_batch_scores.device)
    
    # (batch_size, n_ways, q_seq_len, d_seq_len)
    aligned_scores = cross_batch_scores[batch_indices, batch_indices]
    aligned_is_retrieved = is_retrieved[batch_indices, batch_indices]

    # 6. Compute Max Similarity for Retrieved Tokens
    # We use -inf for non-retrieved tokens so they don't affect the max.
    masked_scores = aligned_scores.masked_fill(~aligned_is_retrieved, -float('inf'))

    # Take max over document tokens (dim -1)
    # (batch_size, n_ways, q_seq_len)
    max_sim_per_query_token = masked_scores.max(dim=-1).values

    # 7. Handle "Nothing Retrieved" Cases
    valid_retrieval_mask = aligned_is_retrieved.any(dim=-1)

    max_sim_per_query_token = torch.where(
        valid_retrieval_mask,
        max_sim_per_query_token,
        torch.zeros_like(max_sim_per_query_token)
    )

    # 8. Compute Normalizer Z and Final Scores
    # Sum over query tokens to get the numerator
    # (batch_size, n_ways)
    numerator = max_sim_per_query_token.sum(dim=-1)
    
    # For logging: compute retrieved counts per query token
    # (batch_size, n_ways, q_seq_len)
    retrieved_counts = aligned_is_retrieved.sum(dim=-1).float()

    if use_normalizer_Z:
        # Z = number of query tokens that retrieved at least one document token
        # (batch_size, n_ways)
        normalizer_Z = valid_retrieval_mask.sum(dim=-1).to(numerator.dtype)
        
        safe_Z = normalizer_Z.clamp(min=Z_clamp_value)
        scores = numerator / safe_Z
        
        xtr_scores = torch.where(normalizer_Z > 0, scores, torch.zeros_like(scores, dtype=scores.dtype))
    else:
        xtr_scores = numerator

    # -------------------------------------------------------------------------
    # WANDB LOGGING
    # -------------------------------------------------------------------------
    
    # Define masks for Positives (index 0) and Negatives (indices 1..n_ways)
    # (batch_size, n_ways)
    pos_mask = torch.zeros((batch_size, n_ways), dtype=torch.bool, device=xtr_scores.device)
    pos_mask[:, 0] = True
    neg_mask = ~pos_mask
    
    # Aggregate retrieved_counts over q_seq_len dimension for logging (mean count per pair)
    # (batch_size, n_ways)
    retrieved_counts_agg = retrieved_counts.mean(dim=-1)
    
    log_dict = {
        "numerator_mean": numerator.mean().item(),
        "numerator_std": numerator.std().item(),
        "numerator_min": numerator.min().item(),
        "numerator_max": numerator.max().item(),
        "xtr_scores_mean": xtr_scores.mean().item(),
        "xtr_scores_std": xtr_scores.std().item(),
        "k_prime": k_prime,
        "retrieved_counts_mean": retrieved_counts.mean().item(),
        "retrieved_counts_std": retrieved_counts.std().item(),
        "retrieved_counts_max": retrieved_counts.max().item(),
        
        # Positive Stats
        "numerator_mean_pos": numerator[pos_mask].mean().item(),
        "numerator_std_pos": numerator[pos_mask].std().item(),
        "numerator_min_pos": numerator[pos_mask].min().item(),
        "numerator_max_pos": numerator[pos_mask].max().item(),
        "xtr_scores_mean_pos": xtr_scores[pos_mask].mean().item(),
        "xtr_scores_std_pos": xtr_scores[pos_mask].std().item(),
        "retrieved_counts_mean_pos": retrieved_counts_agg[pos_mask].mean().item(),
        "retrieved_counts_std_pos": retrieved_counts_agg[pos_mask].std().item(),
        "retrieved_counts_max_pos": retrieved_counts_agg[pos_mask].max().item(),
        
        # Negative Stats
        "numerator_mean_neg": numerator[neg_mask].mean().item(),
        "numerator_std_neg": numerator[neg_mask].std().item(),
        "numerator_min_neg": numerator[neg_mask].min().item(),
        "numerator_max_neg": numerator[neg_mask].max().item(),
        "xtr_scores_mean_neg": xtr_scores[neg_mask].mean().item(),
        "xtr_scores_std_neg": xtr_scores[neg_mask].std().item(),
        "retrieved_counts_mean_neg": retrieved_counts_agg[neg_mask].mean().item(),
        "retrieved_counts_std_neg": retrieved_counts_agg[neg_mask].std().item(),
        "retrieved_counts_max_neg": retrieved_counts_agg[neg_mask].max().item(),
    }
    
    if use_normalizer_Z:
        log_dict.update({
            "Z_mean": normalizer_Z.float().mean().item(),
            "Z_std": normalizer_Z.float().std().item(),
            "Z_min": normalizer_Z.min().item(),
            "Z_max": normalizer_Z.max().item(),
            "Z_mean_pos": normalizer_Z[pos_mask].float().mean().item(),
            "Z_std_pos": normalizer_Z[pos_mask].float().std().item(),
            "Z_min_pos": normalizer_Z[pos_mask].min().item(),
            "Z_max_pos": normalizer_Z[pos_mask].max().item(),
            "Z_mean_neg": normalizer_Z[neg_mask].float().mean().item(),
            "Z_std_neg": normalizer_Z[neg_mask].float().std().item(),
            "Z_min_neg": normalizer_Z[neg_mask].min().item(),
            "Z_max_neg": normalizer_Z[neg_mask].max().item(),
        })

    # Ranking Stats
    # Check if the positive (index 0) has the highest score in the n_ways dimension
    top_idx = xtr_scores.argmax(dim=1)  # (batch_size,)
    frac_pos_top = (top_idx == 0).float().mean().item()
    
    # Check if argmax of numerator aligns with argmax of final score
    same_argmax_frac = (numerator.argmax(dim=1) == top_idx).float().mean().item()

    log_dict.update({
        "frac_pos_top": frac_pos_top,
        "same_argmax_frac": same_argmax_frac,
        "k_index": k_index,
    })

    if is_main_process():
        wandb.log(log_dict)

    return xtr_scores


class ScheduledXTRScore:
    """Callable wrapper for XTR score functions with scheduled k_prime.
    
    This class allows k_prime to be annealed during training based on the current
    training step. The scheduler function should take the current step as input
    and return the k_prime value to use. Works with both contrastive and KD score functions.
    
    Parameters
    ----------
    score_fn
        The XTR score function to wrap. Must accept (queries_embeddings, documents_embeddings,
        queries_mask, documents_mask, k_prime, use_normalizer_Z, Z_clamp_value).
        Examples: xtr_contrastive_training_scores, xtr_kd_training_scores
    k_prime_scheduler
        Callable that takes the current training step (int) and returns the k_prime (int)
        to use at that step. Example: lambda step: min(100, 10 + step // 100)
    use_normalizer_Z
        Whether to use the normalizer Z in the score computation.
    Z_clamp_value
        Minimum value to clamp Z to prevent division by zero.
    start_normalizer_Z_at_step
        Step at which to start using the normalizer Z.
    
    Examples
    --------
    >>> from pylate.scores import ScheduledXTRScore, xtr_contrastive_training_scores
    >>> 
    >>> # Linear annealing from 10 to 100 over 10000 steps
    >>> def k_prime_scheduler(step: int) -> int:
    ...     k_prime_start = 10
    ...     k_prime_end = 100
    ...     total_steps = 10000
    ...     if step >= total_steps:
    ...         return k_prime_end
    ...     progress = step / total_steps
    ...     return int(k_prime_start + (k_prime_end - k_prime_start) * progress)
    >>> 
    >>> scheduled_score = ScheduledXTRScore(
    ...     score_fn=xtr_contrastive_training_scores,
    ...     k_prime_scheduler=k_prime_scheduler,
    ...     use_normalizer_Z=False,
    ... )
    >>> 
    >>> # Update step before each forward pass (done via callback)
    >>> scheduled_score.update_step(5000)
    >>> 
    >>> # Use as score_metric in loss functions
    >>> # train_loss = losses.Contrastive(model=model, score_metric=scheduled_score)
    >>> # train_loss = losses.Distillation(model=model, score_metric=scheduled_score)
    """
    
    def __init__(
        self,
        score_fn,  # Callable that accepts XTR score function signature
        k_prime_scheduler,  # Callable[[int], int]
        use_normalizer_Z: bool = False,
        Z_clamp_value: float = 1.0,
        start_normalizer_Z_at_step: int = 0,
        impute_scores_instead_of_zero: bool = False,
    ):
        self.score_fn = score_fn
        self.k_prime_scheduler = k_prime_scheduler
        self.use_normalizer_Z = use_normalizer_Z
        self.Z_clamp_value = Z_clamp_value
        self.start_normalizer_Z_at_step = start_normalizer_Z_at_step
        self.impute_scores_instead_of_zero = impute_scores_instead_of_zero
        self.current_step = 0
        self.current_k_prime = None
    
    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute XTR scores with scheduled k_prime.
        
        Parameters
        ----------
        queries_embeddings
            Query embeddings. Shape: (batch_size, num_tokens_queries, embedding_size)
        documents_embeddings
            Document embeddings. Shape: (batch_size, num_tokens_documents, embedding_size) for contrastive
            or (batch_size, n_ways, num_tokens_documents, embedding_size) for KD
        queries_mask
            Mask for query embeddings. Shape: (batch_size, num_tokens_queries)
        documents_mask
            Mask for document embeddings. Shape: (batch_size, num_tokens_documents) for contrastive
            or (batch_size, n_ways, num_tokens_documents) for KD
        
        Returns
        -------
        scores
            XTR scores. Shape: (batch_size, batch_size) for contrastive or (batch_size, n_ways) for KD
        """
        # Compute current k_prime based on step
        self.current_k_prime = self.k_prime_scheduler(self.current_step)
        
        # Determine whether to use normalizer Z based on current step
        should_use_normalizer_Z = (
            self.use_normalizer_Z 
            and self.current_step >= self.start_normalizer_Z_at_step
        )
        
        if should_use_normalizer_Z and self.current_step == self.start_normalizer_Z_at_step:
            print(f"Starting normalizer Z at step {self.current_step}")
        
        # Call the wrapped score function with scheduled k_prime
        return self.score_fn(
            queries_embeddings=queries_embeddings,
            documents_embeddings=documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
            k_prime=int(self.current_k_prime),  # Ensure it's an integer
            use_normalizer_Z=should_use_normalizer_Z,
            Z_clamp_value=self.Z_clamp_value,
            impute_scores_instead_of_zero=self.impute_scores_instead_of_zero,
        )
    
    def update_step(self, step: int):
        """Update the current training step.
        
        Parameters
        ----------
        step
            Current global training step.
        """
        self.current_step = step

class KPrimeSchedulerCallback(TrainerCallback):
    """Callback to update k_prime scheduler with current training step.
    
    This callback should be added to the trainer when using ScheduledXTRScore
    to ensure the k_prime value is updated based on the current training step.
    
    Examples
    --------
    >>> from pylate.scores import ScheduledXTRScore, KPrimeSchedulerCallback
    >>> from pylate.scores import xtr_contrastive_training_scores
    >>> 
    >>> def k_prime_scheduler(step: int) -> int:
    ...     return min(100, 10 + step // 100)
    >>> 
    >>> scheduled_score = ScheduledXTRScore(
    ...     score_fn=xtr_contrastive_training_scores,
    ...     k_prime_scheduler=k_prime_scheduler,
    ... )
    >>> 
    >>> # Add to trainer
    >>> trainer.add_callback(KPrimeSchedulerCallback(scheduled_score))
    """
    
    def __init__(self, scheduled_score_fn):
        """Initialize the callback.
        
        Parameters
        ----------
        scheduled_score_fn
            The ScheduledXTRScore instance to update.
        """
        self.scheduled_score_fn = scheduled_score_fn
    
    def on_step_end(self, args, state, control, **kwargs):
        """Update the step in the scheduled score function.
        
        Parameters
        ----------
        args
            Training arguments.
        state
            Training state containing global_step.
        control
            Training control object.
        
        Returns
        -------
        control
            The training control object.
        """
        if hasattr(self.scheduled_score_fn, 'update_step'):
            self.scheduled_score_fn.update_step(state.global_step)
        return control
