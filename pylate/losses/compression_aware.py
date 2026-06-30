from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from ..models.compression import Compressor, build_compressor
from .contrastive import extract_skiplist_mask


class CompressionAwareLoss(nn.Module):
    """Mix a base multi-vector loss over raw and compressed embeddings.

    Wraps any base loss that exposes :meth:`embed` and
    :meth:`loss_from_embeddings` (e.g. :class:`~pylate.losses.Contrastive`). The
    model is run **once** to produce the full-precision token embeddings; the
    *same* base loss is then applied twice -- to the raw embeddings and to a
    compressed copy (quantized and/or pooled per side) -- and the two scalars are
    combined as ``lam * full + (1 - lam) * compressed``.

    Because compression is derived from the already-computed embeddings (a
    same-shape straight-through quantizer and/or a differentiable token pool), no
    second forward pass is needed and gradients from both terms flow back to the
    full-precision weights. This is the straight-through estimator applied at the
    loss level: the network is told "here is the retrieval loss your embeddings
    incur, and here is the loss they would incur once compressed," and learns to
    arrange embeddings that survive the chosen compression.

    Train with a **vanilla** model -- do not also append a
    :class:`~pylate.models.StraightThroughEstimator` module, or the embeddings
    would be compressed twice. The compressors here describe the target
    query/document compression for training; apply the matching compression
    (quantizer module and/or index-time pooling) at inference.

    Parameters
    ----------
    loss
        Base loss instance sharing the ColBERT model (e.g. ``Contrastive(model)``).
    query_compressor
        Compression applied to the query column. Defaults to the identity
        (queries left uncompressed).
    document_compressor
        Compression applied to every document column. Defaults to the identity.
    lam
        Weight in ``[0, 1]`` on the full-precision loss. ``1.0`` recovers the
        plain base loss; ``0.0`` trains purely on the compressed loss.

    Examples
    --------
    >>> from pylate import models, losses
    >>> from pylate.models import CastQuantizer, SHBQQuantizer
    >>> from pylate.models.compression import Compressor
    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )
    >>> loss = losses.CompressionAwareLoss(
    ...     loss=losses.Contrastive(model=model),
    ...     query_compressor=Compressor(quantizer=CastQuantizer("int8")),
    ...     document_compressor=Compressor(quantizer=SHBQQuantizer()),
    ...     lam=0.5,
    ... )
    >>> anchor = model.preprocess(["fruits are healthy."], is_query=True)
    >>> positive = model.preprocess(["fruits are good for health."], is_query=False)
    >>> negative = model.preprocess(["fruits are bad for health."], is_query=False)
    >>> value = loss([anchor, positive, negative])
    >>> assert isinstance(value.item(), float)

    """

    def __init__(
        self,
        loss: nn.Module,
        query_compressor: Compressor | None = None,
        document_compressor: Compressor | None = None,
        lam: float = 0.5,
    ) -> None:
        super().__init__()
        for required in ("embed", "loss_from_embeddings"):
            if not callable(getattr(loss, required, None)):
                raise TypeError(
                    f"{type(loss).__name__} does not support compression-aware "
                    f"training: it must expose a callable '{required}'. "
                    "Contrastive is supported."
                )
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}.")
        self.loss = loss
        self.model = loss.model
        self.query_compressor = query_compressor or Compressor()
        self.document_compressor = document_compressor or Compressor()
        self.lam = lam

    def _skiplist(self) -> list[int]:
        # Unwrap (D)DP to reach the model attributes, mirroring the base loss.
        model = self.model if hasattr(self.model, "skiplist") else self.model.module
        return model.skiplist

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sentence_features = list(sentence_features)

        # One model pass shared by both loss terms.
        embeddings = self.loss.embed(sentence_features)

        full_loss = self.loss.loss_from_embeddings(
            embeddings=embeddings,
            sentence_features=sentence_features,
            labels=labels,
        )

        # Skip the compressed term entirely when it carries no weight and the
        # pipelines are trivial -- avoids needless scoring on the lam == 1 path.
        if self.lam >= 1.0 and (
            self.query_compressor.is_identity and self.document_compressor.is_identity
        ):
            return full_loss

        base_masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=self._skiplist()
        )
        compressed_embeddings: list[torch.Tensor] = []
        compressed_masks: list[torch.Tensor] = []
        for index, (embedding, mask) in enumerate(zip(embeddings, base_masks)):
            compressor = (
                self.query_compressor if index == 0 else self.document_compressor
            )
            compressed_embedding, compressed_mask = compressor(embedding, mask)
            compressed_embeddings.append(compressed_embedding)
            compressed_masks.append(compressed_mask)

        compressed_loss = self.loss.loss_from_embeddings(
            embeddings=compressed_embeddings,
            sentence_features=sentence_features,
            labels=labels,
            masks=compressed_masks,
        )

        return self.lam * full_loss + (1.0 - self.lam) * compressed_loss


def build_compression_aware_loss(
    base_loss: nn.Module, spec: dict | None
) -> CompressionAwareLoss:
    """Build a :class:`CompressionAwareLoss` from a config block.

    ``spec`` is the ``compression:`` mapping used by the training runner::

        {"lambda": 0.5,
         "query":    {"quantizer": {...}, "pooler": {...}},
         "document": {"quantizer": {...}, "pooler": {...}}}

    The per-side ``query`` / ``document`` blocks are passed to
    :func:`pylate.models.build_compressor`, so the same config atoms drive both
    training and index-time compression.
    """
    spec = spec or {}
    return CompressionAwareLoss(
        loss=base_loss,
        query_compressor=build_compressor(spec.get("query")),
        document_compressor=build_compressor(spec.get("document")),
        lam=float(spec.get("lambda", 0.5)),
    )
