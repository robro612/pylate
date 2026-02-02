@deprecated("This loss function doesn't work well for some reason")
def xtr_contrastive_training_scores_primeqa(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    k_prime: int = 100,
    use_normalizer_Z: bool = False,
    impute_scores_instead_of_zero: bool = False,
    Z_clamp_value: float = 1.0,
    log_frequency: int = 1,
) -> torch.Tensor:
    """Computes the XTR scores for Contrastive Learning with PrimeQA dataset.

    Parameters
    ----------
    queries_embeddings
        Shape: (batch_size, q_seq_len, embedding_size)
    documents_embeddings
        Shape: (batch_size, d_seq_len, embedding_size)
    queries_mask
        Shape: (batch_size, q_seq_len)
    documents_mask
        Shape: (batch_size, d_seq_len)
    k_prime
        The number of top tokens to consider for each query token.
    use_normalizer_Z
        Whether to use the normalizer Z in the score computation.
    impute_scores_instead_of_zero
        Whether to impute scores instead of zeroing them out (not currently used in this implementation).
    Z_clamp_value
        Minimum value to clamp Z to prevent division by zero.

    Returns
    -------
    scores
        Shape: (batch_size, batch_size)
    """
    #inner product b/w doc and query token embeddings
    Q = queries_embeddings
    D = documents_embeddings

    scores = Q.unsqueeze(1) @ D.transpose(1, 2).unsqueeze(0)  # bxqxdxs

    if documents_mask is not None:
        D_mask = documents_mask.repeat(queries_embeddings.size(0), 1, 1)
        # replace Doc <pad> scores with a large -ve number
        scores.transpose(2, 3)[~D_mask.bool()] = -99999

    ##Qb, Db, Qt = max_scores.shape[:3]
    Qb, Db, Qt, Dt = scores.shape

    clubbed_doc_scores = scores.permute(0, 2, 1, 3).flatten(2, 3)

    topk_scores, topk_indices = clubbed_doc_scores.topk(k_prime, -1)

    # create a boolen vector of True for all positions
    alignment_mask = torch.ones_like(clubbed_doc_scores, dtype=torch.bool)

    # mask Query <pad> scores and indices
    if queries_mask is not None:
        topk_scores = topk_scores * queries_mask.unsqueeze(2)

    # mask the topk positions to 0
    alignment_mask.scatter_(-1, topk_indices, 0)

    # change to 0 all the non-topk position scores, leaving topk scores intact
    clubbed_doc_scores.masked_fill(alignment_mask, 0)

    # change the clubbed scores to original shape of QbxQtxDbxDt
    topk_scores_max = clubbed_doc_scores.view(Qb, Qt, Db, -1).max(-1).values

    # get the normalizer for each doc score as the number of non-zeros scores per doc
    # (batch_size, q_seq_len, batch_size) -> (batch_size, batch_size)
    retrieved_counts = (topk_scores_max > 0.0).float().sum(1)

    # Compute Z (clamped version for normalization)
    Z = retrieved_counts.clamp(min=Z_clamp_value)

    # normalize scores
    numerator = topk_scores_max.sum(1)

    if use_normalizer_Z:
        xtr_scores = (1 / Z) * numerator
    else:
        xtr_scores = numerator

    # log the stats for debugging / monitoring
    # Create masks for positives (diagonals) and negatives (off-diagonals)
    batch_size = numerator.shape[0]
    diag_mask = torch.eye(batch_size, device=numerator.device, dtype=torch.bool)
    off_diag_mask = ~diag_mask

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
        "retrieved_counts_mean_pos": retrieved_counts[diag_mask].mean().item(),
        "retrieved_counts_std_pos": retrieved_counts[diag_mask].std().item(),
        "retrieved_counts_max_pos": retrieved_counts[diag_mask].max().item(),
        "numerator_mean_neg": numerator[off_diag_mask].mean().item(),
        "numerator_std_neg": numerator[off_diag_mask].std().item(),
        "numerator_min_neg": numerator[off_diag_mask].min().item(),
        "numerator_max_neg": numerator[off_diag_mask].max().item(),
        "xtr_scores_mean_neg": xtr_scores[off_diag_mask].mean().item(),
        "xtr_scores_std_neg": xtr_scores[off_diag_mask].std().item(),
        "retrieved_counts_mean_neg": retrieved_counts[off_diag_mask].mean().item(),
        "retrieved_counts_std_neg": retrieved_counts[off_diag_mask].std().item(),
        "retrieved_counts_max_neg": retrieved_counts[off_diag_mask].max().item(),
    }

    # Add Z statistics (all pairs, positives, and negatives) if normalizer is used
    if use_normalizer_Z:
        log_dict.update(
            {
                "Z_mean": retrieved_counts.float().mean().item(),
                "Z_std": retrieved_counts.float().std().item(),
                "Z_min": retrieved_counts.min().item(),
                "Z_max": retrieved_counts.max().item(),
                "Z_mean_pos": retrieved_counts[diag_mask].float().mean().item(),
                "Z_std_pos": retrieved_counts[diag_mask].float().std().item(),
                "Z_min_pos": retrieved_counts[diag_mask].min().item(),
                "Z_max_pos": retrieved_counts[diag_mask].max().item(),
                "Z_mean_neg": retrieved_counts[off_diag_mask].float().mean().item(),
                "Z_std_neg": retrieved_counts[off_diag_mask].float().std().item(),
                "Z_min_neg": retrieved_counts[off_diag_mask].min().item(),
                "Z_max_neg": retrieved_counts[off_diag_mask].max().item(),
            }
        )

        # Compute Spearman correlation between numerator and xtr_scores
        numerator_np = numerator.detach().cpu().numpy()
        xtr_scores_np = xtr_scores.detach().cpu().numpy()

        # Compute correlation for each query (row)
        batch_size = numerator_np.shape[0]
        rank_correlations = []
        for i in range(batch_size):
            corr, _ = spearmanr(numerator_np[i], xtr_scores_np[i])
            rank_correlations.append(corr)

        mean_rank_correlation = (
            sum(rank_correlations) / len(rank_correlations) if rank_correlations else 0.0
        )
        log_dict["Z_rank_correlation"] = mean_rank_correlation

    # log whether Z changes the max document
    # xtr_scores is (batch_size, batch_size)
    top_idx = xtr_scores.argmax(dim=1)  # which doc is top for each query
    pos_idx = torch.arange(xtr_scores.size(0), device=xtr_scores.device)
    frac_pos_top = (top_idx == pos_idx).float().mean().item()
    # how many queries have same argmax?
    same_argmax_frac = (numerator.argmax(dim=1) == xtr_scores.argmax(dim=1)).float().mean().item()

    log_dict.update(
        {
            "frac_pos_top": frac_pos_top,
            "same_argmax_frac": same_argmax_frac,
        }
    )

    if wandb.run is not None and is_main_process() and _should_log("primeqa", log_frequency):
        wandb.log(log_dict)

    return xtr_scores

@deprecated("This does not accept multiple negatives, use xtr_contrastive_training_scores_multiple_negatives instead")
def xtr_contrastive_training_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    k_prime: int = 100,
    use_normalizer_Z: bool = False,
    impute_scores_instead_of_zero: bool = False,
    Z_clamp_value: float = 1.0,
    log_frequency: int = 1,
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
        masked_scores = cross_batch_scores.masked_fill(~is_retrieved, -float("inf"))

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
        torch.zeros_like(max_sim_per_query_token),
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
        # is this necessary?
        # xtr_scores = torch.where(normalizer_Z > 0, scores, torch.zeros_like(scores, dtype=scores.dtype))
        xtr_scores = scores
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
        log_dict.update(
            {
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
            }
        )

        # Compute Spearman correlation between numerator and Z-normalized scores
        numerator_np = numerator.detach().cpu().numpy()
        xtr_scores_np = xtr_scores.detach().cpu().numpy()

        batch_size = numerator_np.shape[0]
        rank_correlations = []
        for i in range(batch_size):
            corr, _ = spearmanr(numerator_np[i], xtr_scores_np[i])
            rank_correlations.append(corr)

        mean_rank_correlation = (
            sum(rank_correlations) / len(rank_correlations) if rank_correlations else 0.0
        )
        log_dict["Z_rank_correlation"] = mean_rank_correlation

    # log whether Z changes the max document
    # xtr_scores is (bq, bd)
    top_idx = xtr_scores.argmax(dim=1)  # which doc is top for each query
    pos_idx = torch.arange(xtr_scores.size(0), device=xtr_scores.device)
    frac_pos_top = (top_idx == pos_idx).float().mean().item()
    # how many queries have same argmax?
    same_argmax_frac = (numerator.argmax(dim=1) == xtr_scores.argmax(dim=1)).float().mean().item()

    log_dict.update(
        {
            "frac_pos_top": frac_pos_top,
            "same_argmax_frac": same_argmax_frac,
            "k_index": k_index,
        }
    )

    if wandb.run is not None and is_main_process() and _should_log("contrastive", log_frequency):
        wandb.log(log_dict)

    return xtr_scores
