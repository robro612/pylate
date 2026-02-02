from __future__ import annotations

import numpy as np
import torch
import wandb
from scipy.stats import spearmanr

from ..utils.tensor import convert_to_tensor


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
    flat_documents_embeddings = documents_embeddings.reshape(
        -1, d_seq_len, documents_embeddings.size(-1)
    )

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
    thresholds = (
        cross_batch_scores_flattened.kthvalue(k=k_index, dim=-1)
        .values.unsqueeze(1)
        .unsqueeze(-1)
    )

    # 5. Determine Retrieval (Alignment Matrix A)
    # (batch_size, total_docs, q_seq_len, d_seq_len)
    is_retrieved = cross_batch_scores >= thresholds

    # Reshape back to separate batch and n_ways
    # (batch_size, batch_size, n_ways, q_seq_len, d_seq_len)
    cross_batch_scores = cross_batch_scores.view(
        batch_size, batch_size, n_ways, q_seq_len, d_seq_len
    )
    is_retrieved = is_retrieved.view(
        batch_size, batch_size, n_ways, q_seq_len, d_seq_len
    )

    # Select only the aligned pairs: Query[i] vs Docs[i]
    batch_indices = torch.arange(batch_size, device=cross_batch_scores.device)

    # (batch_size, n_ways, q_seq_len, d_seq_len)
    aligned_scores = cross_batch_scores[batch_indices, batch_indices]
    aligned_is_retrieved = is_retrieved[batch_indices, batch_indices]

    # 6. Compute Max Similarity for Retrieved Tokens
    # We use -inf for non-retrieved tokens so they don't affect the max.
    masked_scores = aligned_scores.masked_fill(~aligned_is_retrieved, -float("inf"))

    # Take max over document tokens (dim -1)
    # (batch_size, n_ways, q_seq_len)
    max_sim_per_query_token = masked_scores.max(dim=-1).values

    # 7. Handle "Nothing Retrieved" Cases
    valid_retrieval_mask = aligned_is_retrieved.any(dim=-1)

    max_sim_per_query_token = torch.where(
        valid_retrieval_mask,
        max_sim_per_query_token,
        torch.zeros_like(max_sim_per_query_token),
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

        xtr_scores = torch.where(
            normalizer_Z > 0, scores, torch.zeros_like(scores, dtype=scores.dtype)
        )
    else:
        xtr_scores = numerator

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
        log_dict.update(
            {
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
            }
        )

    # Ranking Stats
    # Check if the positive (index 0) has the highest score in the n_ways dimension
    top_idx = xtr_scores.argmax(dim=1)  # (batch_size,)
    frac_pos_top = (top_idx == 0).float().mean().item()

    # Check if argmax of numerator aligns with argmax of final score
    same_argmax_frac = (numerator.argmax(dim=1) == top_idx).float().mean().item()

    # Convert to numpy for scipy
    numerator_np = numerator.detach().cpu().numpy()
    xtr_scores_np = xtr_scores.detach().cpu().numpy()

    # Compute Spearman correlation for each batch element
    batch_size = numerator_np.shape[0]
    rank_correlations = []
    for i in range(batch_size):
        corr, _ = spearmanr(numerator_np[i], xtr_scores_np[i])
        rank_correlations.append(corr)

    mean_rank_correlation = sum(rank_correlations) / len(rank_correlations)

    log_dict.update(
        {
            "frac_pos_top": frac_pos_top,
            "same_argmax_frac": same_argmax_frac,
            "Z_rank_correlation": mean_rank_correlation,
            "k_index": k_index,
        }
    )

    if wandb.run is not None:
        wandb.log(log_dict)

    return xtr_scores
