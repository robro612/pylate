from __future__ import annotations
from typing_extensions import deprecated

import numpy as np
import torch
import wandb
from scipy.stats import spearmanr

from ..utils.tensor import convert_to_tensor


def _should_log(log_frequency: int) -> bool:
    if log_frequency <= 1:
        return True
    if wandb.run is None:
        return False
    step = wandb.run.step or 0
    return (step % log_frequency) == 0


def _register_grad_logging(
    name: str, tensor: torch.Tensor, log_prefix: str, log_frequency: int
) -> None:
    if wandb.run is None:
        return
    if not tensor.requires_grad:
        return

    def _hook(grad: torch.Tensor) -> None:
        if not _should_log(log_frequency):
            return
        grad = grad.detach()
        log_key = f"{log_prefix}/grad/{name}"
        step = (wandb.run.step or 0) if wandb.run is not None else None
        wandb.log(
            {
                f"{log_key}_norm": grad.norm().item(),
                f"{log_key}_abs_mean": grad.abs().mean().item(),
                f"{log_key}_abs_max": grad.abs().max().item(),
                f"{log_key}_non_finite_frac": (~torch.isfinite(grad)).float().mean().item(),
            },
            step=step,
            commit=False,
        )

    tensor.register_hook(_hook)

def xtr_contrastive_training_scores_multiple_negatives(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    k_prime: int = 100,
    use_normalizer_Z: bool = False,
    impute_scores_instead_of_zero: bool = False,
    Z_clamp_value: float = 1.0,
    log_gradients: bool = False,
    log_prefix: str = "xtr",
    log_z_stats: bool = True,
    log_frequency: int = 10,
    positive_document_index: int = 0,
) -> torch.Tensor:
    """Computes the XTR scores for Contrastive Learning when each query has multiple candidate documents.

    This is the "multiple negatives" variant of `xtr_contrastive_training_scores`.

    Shapes
    ------
    queries_embeddings:
        (batch_queries, q_seq_len, embedding_size)

        (batch_docs, n_docs, d_seq_len, embedding_size)
        where n_docs typically corresponds to [positive, neg_1, ..., neg_k] per query.
    queries_mask:
        (batch_queries, q_seq_len)
    documents_mask:
        (batch_docs, n_docs, d_seq_len)

    Returns
    -------
    scores:
        (batch_queries, batch_docs * n_docs)
        Suitable for cross-entropy where the positive for query i is located at
        column (global_doc_index * n_docs + positive_document_index).

    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    if documents_embeddings.ndim != 4:
        raise ValueError(
            "documents_embeddings must have shape (batch_docs, n_docs, d_seq_len, embedding_size); "
            f"got {tuple(documents_embeddings.shape)}"
        )

    bq, q_seq_len, _ = queries_embeddings.shape
    bd, n_docs, d_seq_len, _ = documents_embeddings.shape
    if bq != bd:
        raise ValueError(
            "queries_embeddings and documents_embeddings must have the same batch size; "
            f"got bq={bq}, bd={bd}."
        )
    if not (0 <= positive_document_index < n_docs):
        raise ValueError(
            "positive_document_index must be in [0, {n_docs - 1}] "
            f"but got {positive_document_index}."
        )

    # Flatten the (batch_docs, n_docs) dims into a single document batch dim.
    # flat_docs: (bd * n_docs, d_seq_len, embedding_size)
    flat_docs = documents_embeddings.reshape(
        bd * n_docs, d_seq_len, documents_embeddings.size(-1)
    )

    # 1. Compute raw Cross-Batch Scores
    # (batch_queries, bd*n_docs, q_seq_len, d_seq_len)
    cross_batch_scores = torch.einsum(
        "aqh, bdh->abqd",
        queries_embeddings,
        flat_docs,
    )

    # 2. Apply Padding Masking
    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        cross_batch_scores = cross_batch_scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)
        if documents_mask.ndim != 3:
            raise ValueError(
                "documents_mask must have shape (batch_docs, n_docs, d_seq_len); "
                f"got {tuple(documents_mask.shape)}"
            )
        flat_docs_mask = documents_mask.reshape(bd * n_docs, d_seq_len)
        cross_batch_scores = cross_batch_scores * flat_docs_mask.unsqueeze(0).unsqueeze(2)

    total_docs = bd * n_docs

    # 3. Flatten to find Global Top-K Thresholds
    # (batch_queries, q_seq_len, total_docs * d_seq_len)
    cross_batch_scores_flattened = cross_batch_scores.permute(0, 2, 1, 3).reshape(
        bq, q_seq_len, total_docs * d_seq_len
    )

    k_index = max(1, cross_batch_scores_flattened.size(-1) - k_prime + 1)
    thresholds = cross_batch_scores_flattened.kthvalue(k=k_index, dim=-1).values
    thresholds = thresholds.unsqueeze(1).unsqueeze(-1)  # (bq, 1, q_len, 1)

    # 4. Determine Retrieval (Alignment Matrix A)
    is_retrieved = cross_batch_scores >= thresholds

    # 5. Compute Max Similarity for Retrieved Tokens
    if impute_scores_instead_of_zero:
        masked_scores = torch.where(is_retrieved, cross_batch_scores, thresholds)
    else:
        masked_scores = cross_batch_scores.masked_fill(~is_retrieved, -float("inf"))

    max_sim_per_query_token = masked_scores.max(dim=-1).values  # (bq, total_docs, q_len)

    # 6. Handle "Nothing Retrieved" Cases
    valid_retrieval_mask = is_retrieved.any(dim=-1)  # (bq, total_docs, q_len)
    max_sim_per_query_token = torch.where(
        valid_retrieval_mask,
        max_sim_per_query_token,
        torch.zeros_like(max_sim_per_query_token),
    )

    # 7. Compute Normalizer Z and Final Scores
    numerator = max_sim_per_query_token.sum(dim=-1)  # (bq, total_docs)

    normalizer_Z = valid_retrieval_mask.sum(dim=-1).to(numerator.dtype)  # (bq, total_docs)
    if use_normalizer_Z:
        xtr_scores = numerator / normalizer_Z.clamp(min=Z_clamp_value)
    else:
        xtr_scores = numerator

    if log_gradients:
        _register_grad_logging(
            "multiple_negatives/numerator", numerator, log_prefix, log_frequency
        )
        _register_grad_logging(
            "multiple_negatives/xtr_scores", xtr_scores, log_prefix, log_frequency
        )

    if log_z_stats and wandb.run is not None and _should_log(log_frequency):
        zero_mask = normalizer_Z == 0
        step = (wandb.run.step or 0) if wandb.run is not None else None
        pos_indices = (
            torch.arange(bq, device=xtr_scores.device) * n_docs + positive_document_index
        )
        pos_doc_scores = xtr_scores.gather(1, pos_indices[:, None]).squeeze(1)
        neg_mask = torch.ones_like(xtr_scores, dtype=torch.bool)
        neg_mask.scatter_(1, pos_indices[:, None], False)
        neg_doc_scores = xtr_scores[neg_mask]
        token_level_scores = max_sim_per_query_token.mean(dim=-1)
        pos_token_scores = token_level_scores.gather(1, pos_indices[:, None]).squeeze(1)
        neg_token_scores = token_level_scores[neg_mask]
        neg_doc_mean = neg_doc_scores.mean().item() if neg_doc_scores.numel() > 0 else 0.0
        neg_token_mean = (
            neg_token_scores.mean().item() if neg_token_scores.numel() > 0 else 0.0
        )
        wandb.log(
            {
                "xtr/normalizer_Z_raw_mean": normalizer_Z.mean().item(),
                "xtr/normalizer_Z_raw_min": normalizer_Z.min().item(),
                "xtr/normalizer_Z_raw_max": normalizer_Z.max().item(),
                "xtr/normalizer_Z_zero_count": zero_mask.sum().item(),
                "xtr/normalizer_Z_zero_frac": zero_mask.float().mean().item(),
                "xtr/doc_score_pos_mean": pos_doc_scores.mean().item(),
                "xtr/doc_score_neg_mean": neg_doc_mean,
                "xtr/token_score_pos_mean": pos_token_scores.mean().item(),
                "xtr/token_score_neg_mean": neg_token_mean,
            },
            step=step,
            commit=False,
        )

    return xtr_scores
