from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import wandb
except ImportError:
    wandb = None

from ..models import ColBERT


def _is_main_process() -> bool:
    return (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or torch.distributed.get_rank() == 0
    )


def _should_log(log_frequency: int) -> bool:
    """Return True when we should log this step (stateless: uses wandb.run.step)."""
    if log_frequency == 0 or wandb is None or wandb.run is None:
        return False
    elif log_frequency == 1:
        return True
    else:
        step = wandb.run.step or 0
        return (step % log_frequency) == 0


class XTRPrimeQA(nn.Module):
    """XTR loss matching the PrimeQA implementation steps."""

    def __init__(
        self,
        model: ColBERT,
        k: int = 55,
        size_average: bool = True,
        log_frequency: int = 25,
    ) -> None:
        super().__init__()
        self.model = model
        self.k = k
        self.size_average = size_average
        self.log_frequency = log_frequency

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the XTR PrimeQA loss.

        Parameters
        ----------
        sentence_features
            List of tokenized sentences. The first sentence is the query and the rest are documents.
        labels
            Unused, present for compatibility with Trainer.
        """
        embeddings = [
            self.model(sentence_feature)["token_embeddings"]
            for sentence_feature in sentence_features
        ]

        query_embeddings = embeddings[0]
        query_attention_mask = sentence_features[0]["attention_mask"]

        documents_embeddings = torch.stack(embeddings[1:], dim=1)
        documents_attention_mask = torch.stack(
            [
                sentence_feature["attention_mask"]
                for sentence_feature in sentence_features[1:]
            ],
            dim=1,
        )

        batch_size, nway = documents_embeddings.shape[:2]
        documents_embeddings = documents_embeddings.view(
            batch_size * nway, *documents_embeddings.shape[2:]
        )
        documents_attention_mask = documents_attention_mask.view(
            batch_size * nway, -1
        )

        # inner product b/w doc and query token embeddings
        scores = query_embeddings.unsqueeze(1) @ documents_embeddings.transpose(
            1, 2
        ).unsqueeze(0)

        D_mask = documents_attention_mask.unsqueeze(0).repeat(
            query_embeddings.size(0), 1, 1
        )

        # replace Doc <pad> scores with a large -ve number
        scores.transpose(2, 3)[~D_mask.bool()] = -99999

        Qb, Db, Qt, Dt = scores.shape

        clubbed_doc_scores = scores.permute(0, 2, 1, 3).flatten(2, 3)

        topk_scores, topk_indices = clubbed_doc_scores.topk(self.k, -1)

        # create a boolean vector of True for all positions
        alignment_mask = torch.ones_like(clubbed_doc_scores, dtype=torch.bool)

        # mask Query <pad> scores and indices
        topk_scores = topk_scores * query_attention_mask.unsqueeze(2)

        # mask the topk positions to 0
        alignment_mask.scatter_(-1, topk_indices, 0)

        # change to 0 all the non-topk position scores, leaving topk scores intact
        # NOTE: the original implementation apparently used `masked_fill` inplace instead of `masked_fill_` which resulted in a no-op and the masking not being done.
        masked_clubbed_doc_scores = clubbed_doc_scores.masked_fill(alignment_mask, 0)

        # change the clubbed scores to original shape of QbxQtxDbxDt
        topk_scores_max = masked_clubbed_doc_scores.view(Qb, Qt, Db, -1).max(-1).values

        # get the normalizer for each doc score as the number of non-zeros scores per doc
        # clamp 0's with some small number to avoid division by zero errors
        Z = (topk_scores_max > 0.0).float().sum(1).clamp(min=1e-3)

        # normalize scores
        doc_tok_summed_normalized = (1 / Z) * topk_scores_max.sum(1)

        # create labels
        labels = torch.arange(
            0, query_embeddings.size(0), device=query_embeddings.device
        ) * nway

        # compute loss
        loss = F.cross_entropy(
            doc_tok_summed_normalized,
            labels,
            reduction="mean" if self.size_average else "sum",
        )

        # wandb: log pos/neg score distributions, Z, and retrieved token pos/neg breakdown
        if (
            wandb is not None
            and wandb.run is not None
            and _is_main_process()
            and _should_log(self.log_frequency)
        ):
            with torch.no_grad():
                Db = batch_size * nway
                Dt = clubbed_doc_scores.shape[-1] // Db
                pos_doc_indices = (
                    torch.arange(Qb, device=query_embeddings.device) * nway
                )

                # Z normalizer stats
                Z_pos = Z[torch.arange(Qb, device=Z.device), pos_doc_indices]
                neg_mask_Z = torch.ones(Qb, Db, dtype=torch.bool, device=Z.device)
                neg_mask_Z.scatter_(1, pos_doc_indices.unsqueeze(1), False)
                Z_neg = Z[neg_mask_Z]
                log_dict = {
                    "xtr_primeqa/Z_mean": Z.mean().item(),
                    "xtr_primeqa/Z_std": Z.std().item(),
                    "xtr_primeqa/Z_min": Z.min().item(),
                    "xtr_primeqa/Z_max": Z.max().item(),
                    "xtr_primeqa/Z_pos_mean": Z_pos.mean().item(),
                    "xtr_primeqa/Z_neg_mean": Z_neg.mean().item(),
                }

                # Doc score distributions (pos vs neg)
                pos_doc_scores = doc_tok_summed_normalized[
                    torch.arange(Qb, device=doc_tok_summed_normalized.device),
                    pos_doc_indices,
                ]
                neg_mask_doc = torch.ones(
                    Qb, Db, dtype=torch.bool, device=doc_tok_summed_normalized.device
                )
                neg_mask_doc.scatter_(1, pos_doc_indices.unsqueeze(1), False)
                neg_doc_scores = doc_tok_summed_normalized[neg_mask_doc]
                log_dict.update({
                    "xtr_primeqa/doc_score_pos_mean": pos_doc_scores.mean().item(),
                    "xtr_primeqa/doc_score_pos_std": pos_doc_scores.std().item(),
                    "xtr_primeqa/doc_score_neg_mean": neg_doc_scores.mean().item(),
                    "xtr_primeqa/doc_score_neg_std": neg_doc_scores.std().item(),
                })

                # Token score distributions (pos vs neg) from topk_scores_max (Qb, Qt, Db)
                pos_token_scores = topk_scores_max[
                    torch.arange(Qb, device=topk_scores_max.device),
                    :,
                    pos_doc_indices,
                ]
                # Neg token scores: all (Qb, Qt, Db) except positive doc column
                neg_token_scores = topk_scores_max.clone()
                neg_token_scores[
                    torch.arange(Qb, device=topk_scores_max.device),
                    :,
                    pos_doc_indices,
                ] = float("nan")
                neg_token_scores = neg_token_scores.reshape(-1)
                neg_token_scores = neg_token_scores[torch.isfinite(neg_token_scores)]
                log_dict.update({
                    "xtr_primeqa/token_score_pos_mean": pos_token_scores.mean().item(),
                    "xtr_primeqa/token_score_pos_std": pos_token_scores.std().item(),
                    "xtr_primeqa/token_score_neg_mean": neg_token_scores.mean().item()
                    if neg_token_scores.numel() > 0
                    else 0.0,
                    "xtr_primeqa/token_score_neg_std": neg_token_scores.std().item()
                    if neg_token_scores.numel() > 0
                    else 0.0,
                })

                # Retrieved token pos/neg breakdown (retrieved = top-k, i.e. alignment_mask == 0)
                # topk_indices: (Qb, Qt, k); each index in [0, Db*Dt); doc_idx = idx // Dt
                doc_idx_retrieved = topk_indices // Dt  # (Qb, Qt, k)
                retrieved_pos_mask = (
                    doc_idx_retrieved
                    == pos_doc_indices.view(Qb, 1, 1)
                )
                q_mask = query_attention_mask.unsqueeze(-1).expand_as(
                    doc_idx_retrieved
                )
                
                # Global counts (across all queries)
                total_retrieved = (q_mask.float()).sum().item()
                num_retrieved_pos = (retrieved_pos_mask.float() * q_mask.float()).sum().item()
                num_retrieved_neg = total_retrieved - num_retrieved_pos
                
                # Per-query percentages (then averaged)
                # For each query: compute percentage, then average across queries
                total_retrieved_per_query = q_mask.float().sum(dim=(1, 2))  # (Qb,)
                num_retrieved_pos_per_query = (
                    retrieved_pos_mask.float() * q_mask.float()
                ).sum(dim=(1, 2))  # (Qb,)
                num_retrieved_neg_per_query = (
                    total_retrieved_per_query - num_retrieved_pos_per_query
                )
                
                # Per-query percentages
                pct_pos_per_query = (
                    100.0
                    * num_retrieved_pos_per_query
                    / total_retrieved_per_query.clamp(min=1e-6)
                )
                pct_neg_per_query = (
                    100.0
                    * num_retrieved_neg_per_query
                    / total_retrieved_per_query.clamp(min=1e-6)
                )
                
                # Average percentages across queries
                pct_retrieved_that_are_pos_avg = pct_pos_per_query.mean().item()
                pct_retrieved_that_are_neg_avg = pct_neg_per_query.mean().item()
                
                # Global percentages: fraction of *retrieved* slots that are pos/neg (always in [0, 100])
                pct_retrieved_that_are_pos_global = (
                    100.0 * num_retrieved_pos / total_retrieved
                    if total_retrieved > 0
                    else 0.0
                )
                pct_retrieved_that_are_neg_global = (
                    100.0 * num_retrieved_neg / total_retrieved
                    if total_retrieved > 0
                    else 0.0
                )
                # Clamp to [0, 100]; same positive token can be in multiple slots, but pos+neg = total_retrieved
                pct_retrieved_that_are_pos_global = min(
                    100.0, max(0.0, pct_retrieved_that_are_pos_global)
                )
                pct_retrieved_that_are_neg_global = min(
                    100.0, max(0.0, pct_retrieved_that_are_neg_global)
                )
                
                # Total positive tokens (non-pad) in positive docs
                pos_doc_global = pos_doc_indices
                pos_doc_mask = documents_attention_mask[pos_doc_global]
                total_pos_tokens = pos_doc_mask.sum().item()
                
                # Per-query: what % of positive tokens were retrieved
                num_retrieved_pos_per_query_tensor = num_retrieved_pos_per_query.float()
                total_pos_tokens_per_query = pos_doc_mask.sum(dim=1).float()  # (Qb,)
                pct_positive_tokens_retrieved_per_query = (
                    100.0
                    * num_retrieved_pos_per_query_tensor
                    / total_pos_tokens_per_query.clamp(min=1e-6)
                )
                pct_positive_tokens_retrieved_avg = (
                    pct_positive_tokens_retrieved_per_query.mean().item()
                )
                
                # Global: retrieval_slots_from_pos / unique_positive_tokens (can exceed 100%: same token retrieved by many query tokens)
                pct_positive_tokens_retrieved_global = (
                    100.0 * num_retrieved_pos / total_pos_tokens
                    if total_pos_tokens > 0
                    else 0.0
                )

                log_dict.update({
                    "xtr_primeqa/retrieved_total": total_retrieved,
                    "xtr_primeqa/retrieved_pos_count": num_retrieved_pos,
                    "xtr_primeqa/retrieved_neg_count": num_retrieved_neg,
                    "xtr_primeqa/pct_retrieved_tokens_pos_avg_per_query": pct_retrieved_that_are_pos_avg,
                    "xtr_primeqa/pct_retrieved_tokens_neg_avg_per_query": pct_retrieved_that_are_neg_avg,
                    "xtr_primeqa/pct_retrieved_tokens_pos_global": pct_retrieved_that_are_pos_global,
                    "xtr_primeqa/pct_retrieved_tokens_neg_global": pct_retrieved_that_are_neg_global,
                    "xtr_primeqa/total_positive_tokens": total_pos_tokens,
                    "xtr_primeqa/pct_positive_tokens_retrieved_avg_per_query": pct_positive_tokens_retrieved_avg,
                    "xtr_primeqa/pct_positive_tokens_retrieved_global": pct_positive_tokens_retrieved_global,
                })
            wandb.log(log_dict)

        return loss
