from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import wandb

from ..models import ColBERT
from ..scores import colbert_scores
from ..utils import all_gather, all_gather_with_gradients, get_rank, get_world_size


def extract_skiplist_mask(
    sentence_features: Iterable[dict[str, torch.Tensor]],
    skiplist: list[int],
) -> list[torch.Tensor]:
    """Extracts the attention masks from the sentence features. We apply a skiplist mask to the documents.
    We skip the first sentence feature because it is the query.

    Examples
    --------
    >>> import torch

    >>> sentence_features = [
    ...     {
    ...         "input_ids": torch.tensor([[1, 2, 3, 4]]),
    ...         "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    ...     },
    ...     {
    ...         "input_ids": torch.tensor([[1, 2, 3, 4]]),
    ...         "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    ...     },
    ...     {
    ...         "input_ids": torch.tensor([[1, 2, 3, 4]]),
    ...         "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    ...     },
    ... ]

    >>> extract_skiplist_mask(
    ...     sentence_features=sentence_features,
    ...     skiplist=[1, 2, 3],
    ... )
    [tensor([[True, True, True, True]]), tensor([[False, False, False,  True]]), tensor([[False, False, False,  True]])]

    """
    attention_masks = [
        sentence_feature["attention_mask"] for sentence_feature in sentence_features
    ]

    skiplist_masks = [
        torch.ones_like(sentence_features[0]["input_ids"], dtype=torch.bool)
    ]

    # We skip the first sentence feature because it is the query.
    skiplist_masks.extend(
        [
            ColBERT.skiplist_mask(
                input_ids=sentence_feature["input_ids"], skiplist=skiplist
            )
            for sentence_feature in sentence_features[1:]
        ]
    )

    return [
        torch.logical_and(skiplist_mask, attention_mask)
        for skiplist_mask, attention_mask in zip(skiplist_masks, attention_masks)
    ]


class Contrastive(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

    Parameters
    ----------
    model
        ColBERT model.
    score_metric
        ColBERT scoring function. Defaults to colbert_scores.
    size_average
        Average by the size of the mini-batch.
    gather_across_devices
        Whether to gather the embeddings across devices to have more in batch negatives. We recommend making sure the sampling across GPUs use the same dataset in case of multi-dataset training to make sure the negatives are plausible.

    Examples
    --------
    >>> from pylate import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> loss = losses.Contrastive(model=model)

    >>> anchor = model.tokenize([
    ...     "fruits are healthy.",
    ... ], is_query=True)

    >>> positive = model.tokenize([
    ...     "fruits are good for health.",
    ... ], is_query=False)

    >>> negative = model.tokenize([
    ...     "fruits are bad for health.",
    ... ], is_query=False)

    >>> sentence_features = [anchor, positive, negative]

    >>> loss = loss(sentence_features=sentence_features)
    >>> assert isinstance(loss.item(), float)

    """

    def __init__(
        self,
        model: ColBERT,
        score_metric : Callable | list[tuple[Callable, float]]=colbert_scores,
        size_average: bool = True,
        gather_across_devices: bool = False,
        temperature: float = 1.0,
        do_auxiliary_loss: None | tuple[int, float] = None,
        score_all_docs_at_once: bool = False,
        positive_document_index: int = 0,
    ) -> None:
        super(Contrastive, self).__init__()
        if isinstance(score_metric, list):
            # normalize the weights so that they sum to 1
            weights_sum = sum(weight for _, weight in score_metric)
            self.score_metrics = [(score_metric, weight / weights_sum) for score_metric, weight in score_metric]
        else:
            self.score_metrics = [(score_metric, 1.0)]
        self.model = model
        self.size_average = size_average
        self.gather_across_devices = gather_across_devices
        self.temperature = temperature
        self.score_all_docs_at_once = score_all_docs_at_once
        self.positive_document_index = positive_document_index
        if do_auxiliary_loss is not None:
            self.do_auxiliary_loss = True
            self.kprime = do_auxiliary_loss[0]
            self.auxiliary_loss_weight = do_auxiliary_loss[1]
        else:
            self.do_auxiliary_loss = False


    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Constrastive loss.

        Parameters
        ----------
        sentence_features
            List of tokenized sentences. The first sentence is the anchor and the rest are the positive and negative examples.
        labels
            The labels for the contrastive loss. Not used in this implementation, but kept for compatibility with Trainer.

        """
        embeddings = [
            torch.nn.functional.normalize(
                self.model(sentence_feature)["token_embeddings"], p=2, dim=-1
            )
            for sentence_feature in sentence_features
        ]

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
        batch_size = embeddings[0].size(0)
        rank = get_rank() if self.gather_across_devices else 0

        if self.score_all_docs_at_once:
            if len(embeddings) < 2:
                raise ValueError(
                    "Contrastive(score_all_docs_at_once=True) expects at least 2 sentence_features: "
                    "[query, documents...]"
                )
            if not (0 <= self.positive_document_index < (len(embeddings) - 1)):
                raise ValueError(
                    f"positive_document_index must be in [0, {len(embeddings) - 2}] "
                    f"but got {self.positive_document_index}."
                )

            # Stack all document groups into a single tensor so score_metrics can use a shared
            # retrieval pool / thresholding (required for XTR-style scoring).
            # docs_embeddings: (bs, n_docs, d_seq_len, dim)
            docs_embeddings = torch.stack(embeddings[1:], dim=1)
            docs_masks = torch.stack(masks[1:], dim=1)  # (bs, n_docs, d_seq_len)

            # Create corresponding labels: positive document for query i is at column
            # (global_doc_index * n_docs + positive_document_index)
            n_docs = docs_embeddings.size(1)
            labels = (
                torch.arange(0, batch_size, device=embeddings[0].device) * n_docs
                + int(self.positive_document_index)
            )

            # Possibly gather documents across devices to have more in-batch negatives.
            if self.gather_across_devices:
                # Keep gradients for docs embeddings, but not for masks.
                docs_embeddings = torch.cat(
                    all_gather_with_gradients(docs_embeddings), dim=0
                )
                docs_masks = torch.cat(all_gather(docs_masks), dim=0)
                labels = labels + rank * batch_size * n_docs

            losses = []
            for score_metric, weight in self.score_metrics:
                scores = score_metric(
                    embeddings[0],
                    docs_embeddings,
                    queries_mask=masks[0] if not do_query_expansion else None,
                    documents_mask=docs_masks,
                )
                loss = F.cross_entropy(
                    input=scores / self.temperature,
                    target=labels,
                    reduction="mean" if self.size_average else "sum",
                )
                losses.append(weight * loss)

            if wandb.run is not None:
                wandb.log({f"losses_{i}": loss.item() for i, loss in enumerate(losses)})

            loss = sum(losses)

            if self.do_auxiliary_loss:
                # Auxiliary loss currently assumes exactly (query, positive, negative).
                raise ValueError(
                    "do_auxiliary_loss is not compatible with score_all_docs_at_once=True. "
                    "It assumes exactly [query, positive, negative]."
                )

            if self.gather_across_devices:
                loss *= get_world_size()
            return loss

        # Default behavior: score each doc group separately (can be fine for dot-product style scores).

        # create corresponding labels
        labels = torch.arange(0, batch_size, device=embeddings[0].device)
        # Possibly gather the embeddings across devices to have more in batch negatives.
        if self.gather_across_devices:
            # Note that we only gather the documents embeddings and not the queries embeddings (embeddings[0]), but are keeping gradients. This is to lower the memory usage, see https://github.com/mlfoundations/open_clip/issues/616
            embeddings = [
                embeddings[0],
                *[
                    torch.cat(all_gather_with_gradients(embedding))
                    for embedding in embeddings[1:]
                ],
            ]
            # Masks [0] is the anchor mask so we do not need to gather it (even though we are not using it for now anyways)
            # Also, we do gather without gradients for the masks as we do not backpropagate through them
            masks = [
                masks[0],
                *[torch.cat(all_gather(mask)) for mask in masks[1:]],
            ]
            # Adjust the labels to match the gathered embeddings positions
            labels = labels + rank * batch_size
        # Note: the queries mask is not used, if added, take care that the expansion tokens are not masked from scoring (because they might be masked during encoding).
        # We might not need to compute the mask for queries but I let the logic there for now

        losses = []
        for score_metric, weight in self.score_metrics:
            scores = torch.cat(
                [
                    score_metric(
                        embeddings[0],
                        group_embeddings,
                        queries_mask=masks[0] if not do_query_expansion else None,
                        documents_mask=documents_masks,
                    )
                    for group_embeddings, documents_masks in zip(embeddings[1:], masks[1:])
                ],
                dim=1,
            )

            # compute constrastive loss using cross-entropy over the scores
            loss = F.cross_entropy(
                input=scores / self.temperature,
                target=labels,
                reduction="mean" if self.size_average else "sum",
            )
            losses.append(weight * loss)

        if wandb.run is not None:
            wandb.log({f"losses_{i}": loss.item() for i, loss in enumerate(losses)})
        
        loss = sum(losses)

        if self.do_auxiliary_loss:
            q_embeddings = embeddings[0]
            p_embeddings = embeddings[1] * masks[1][..., None]
            n_embeddings = embeddings[2] * masks[2][..., None]

            p_scores = torch.einsum("bnd, bsd -> bns", q_embeddings, p_embeddings)
            n_scores = torch.einsum("bnd, bsd -> bns", q_embeddings, n_embeddings)

            k = min(self.kprime, p_scores.shape[-1], n_scores.shape[-1])

            p_scores_topk = p_scores.topk(k=k, dim=-1).values
            n_scores_topk = n_scores.topk(k=k, dim=-1).values

            p_scores_topk_sum = p_scores_topk.sum(dim=-1).view(-1)
            n_scores_topk_sum = n_scores_topk.sum(dim=-1).view(-1)

            aux_loss = torch.nn.functional.softplus(n_scores_topk_sum - p_scores_topk_sum).mean()

            wandb.log({
                "contrastive_loss": loss.item(),
                "auxiliary_loss": aux_loss.item(),
                "aux_loss_weight": self.auxiliary_loss_weight,
                "aux_loss_k": k,
            })

            loss = loss + self.auxiliary_loss_weight * aux_loss

        # Scale by world size when gathering across device
        if self.gather_across_devices:
            loss *= get_world_size()
        return loss
