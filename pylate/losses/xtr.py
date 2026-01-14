from __future__ import annotations

from typing import Callable, Iterable

import torch
import wandb

from ..models import ColBERT
from ..scores import colbert_kd_scores
from .contrastive import extract_skiplist_mask


def is_main_process():
    return (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or torch.distributed.get_rank() == 0
    )


class XTR(torch.nn.Module):
    """XTR loss for ColBERT model, Sean from MixedBread's style.
    #TODO: if this works, make the format more comensurate with the usual pylate format.


    Parameters
    ----------
    model
        SentenceTransformer model.
    score_metric
        Function that returns a score between two sequences of embeddings.
    size_average : bool
        Average by the size of the mini-batch or perform sum.
    """

    def __init__(
        self,
        model: ColBERT,
        k_prime: int = 100,
    ) -> None:
        super(XTR, self).__init__()
        self.model = model
        self.k_prime = k_prime
    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor
    ) -> torch.Tensor:
        """
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
        positive_documents_embeddings = torch.nn.functional.normalize(
            self.model(sentence_features[1])["token_embeddings"], p=2, dim=-1
        )

        negative_documents_embeddings = torch.nn.functional.normalize(
            self.model(sentence_features[2])["token_embeddings"], p=2, dim=-1
        )

        # Reshape them to (bs, n_ways)
        positive_documents_embeddings = positive_documents_embeddings.view(
            queries_embeddings.size(0), -1, *positive_documents_embeddings.shape[1:]
        )
        negative_documents_embeddings = negative_documents_embeddings.view(
            queries_embeddings.size(0), -1, *negative_documents_embeddings.shape[1:]
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
        
        if not do_query_expansion:
            queries_embeddings = queries_embeddings * masks[0][..., None]

        positive_documents_embeddings_mask = masks[1].view(
            queries_embeddings.size(0), -1, *masks[1].shape[1:]
        )

        negative_documents_embeddings_mask = masks[2].view(
            queries_embeddings.size(0), -1, *masks[2].shape[1:]
        )

        # print(f"{positive_documents_embeddings.shape=} {positive_documents_embeddings_mask.shape=}")
        # print(f"{negative_documents_embeddings.shape=} {negative_documents_embeddings_mask.shape=}")
        positive_documents_embeddings = positive_documents_embeddings * positive_documents_embeddings_mask[..., None]
        negative_documents_embeddings = negative_documents_embeddings * negative_documents_embeddings_mask[..., None]

        pos_scores = torch.einsum("bnd, bsd -> bns", queries_embeddings, positive_documents_embeddings.squeeze(1))
        neg_scores = torch.einsum("bnd, bsd -> bns", queries_embeddings, negative_documents_embeddings.squeeze(1))

        pos_scores_topk = pos_scores.topk(k=self.k_prime, dim=-1).values
        neg_scores_topk = neg_scores.topk(k=self.k_prime, dim=-1).values

        # Compute sums for loss computation
        pos_scores_topk_sum = pos_scores_topk.sum(dim=-1).view(-1)  # (batch_size,)
        neg_scores_topk_sum = neg_scores_topk.sum(dim=-1).view(-1)  # (batch_size,)
        
        # Compute the difference before softplus
        score_diff = neg_scores_topk_sum - pos_scores_topk_sum
        
        # Compute loss
        loss = torch.nn.functional.softplus(score_diff).mean()

        # Log statistics
        log_dict = {
            # Raw scores statistics
            "pos_scores_mean": pos_scores.mean().item(),
            "pos_scores_std": pos_scores.std().item(),
            "pos_scores_min": pos_scores.min().item(),
            "pos_scores_max": pos_scores.max().item(),
            "neg_scores_mean": neg_scores.mean().item(),
            "neg_scores_std": neg_scores.std().item(),
            "neg_scores_min": neg_scores.min().item(),
            "neg_scores_max": neg_scores.max().item(),
            
            # Top-k scores statistics
            "pos_scores_topk_mean": pos_scores_topk.mean().item(),
            "pos_scores_topk_std": pos_scores_topk.std().item(),
            "pos_scores_topk_min": pos_scores_topk.min().item(),
            "pos_scores_topk_max": pos_scores_topk.max().item(),
            "neg_scores_topk_mean": neg_scores_topk.mean().item(),
            "neg_scores_topk_std": neg_scores_topk.std().item(),
            "neg_scores_topk_min": neg_scores_topk.min().item(),
            "neg_scores_topk_max": neg_scores_topk.max().item(),
            
            # Sum statistics
            "pos_scores_topk_sum_mean": pos_scores_topk_sum.mean().item(),
            "pos_scores_topk_sum_std": pos_scores_topk_sum.std().item(),
            "pos_scores_topk_sum_min": pos_scores_topk_sum.min().item(),
            "pos_scores_topk_sum_max": pos_scores_topk_sum.max().item(),
            "neg_scores_topk_sum_mean": neg_scores_topk_sum.mean().item(),
            "neg_scores_topk_sum_std": neg_scores_topk_sum.std().item(),
            "neg_scores_topk_sum_min": neg_scores_topk_sum.min().item(),
            "neg_scores_topk_sum_max": neg_scores_topk_sum.max().item(),
            
            # Difference statistics
            "score_diff_mean": score_diff.mean().item(),
            "score_diff_std": score_diff.std().item(),
            "score_diff_min": score_diff.min().item(),
            "score_diff_max": score_diff.max().item(),
            
            "k_prime": self.k_prime,
            
            # Margin statistics (how much better positives are than negatives)
            "margin_mean": (pos_scores_topk_sum - neg_scores_topk_sum).mean().item(),
            "margin_std": (pos_scores_topk_sum - neg_scores_topk_sum).std().item(),
            "margin_min": (pos_scores_topk_sum - neg_scores_topk_sum).min().item(),
            "margin_max": (pos_scores_topk_sum - neg_scores_topk_sum).max().item(),

            "frac_pos_better": (pos_scores_topk_sum > neg_scores_topk_sum).float().mean().item(),
        }
        
        if is_main_process():
            wandb.log(log_dict)

        return loss