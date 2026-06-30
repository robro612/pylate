"""Sequence-length compression (token pooling) for compression-aware training.

This complements :mod:`pylate.models.quantization`. Where a :class:`Quantizer`
maps embeddings to a coarser *value* grid at a fixed token count, a :class:`Pooler`
reduces the *number* of tokens by clustering and mean-pooling them -- the
seq-len axis of multi-vector compression.

Both are combined by a :class:`Compressor`, the per-side (query/document)
pipeline consumed by :class:`~pylate.losses.CompressionAwareLoss`: it optionally
pools, then optionally quantizes, the token embeddings the loss scores.

Pooling is made differentiable for straight-through training: the (discrete,
non-differentiable) cluster *assignment* is computed under ``no_grad`` and treated
as a constant, while the mean within each cluster is a plain differentiable
average, so gradients flow back to the full-precision token embeddings. The model
therefore learns to arrange its tokens so that pooling preserves what matters --
exactly the straight-through estimator trick, applied to the assignment instead
of to rounding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .quantization import (
    Quantizer,
    _friendly_quantizer_spec,
    build_quantizer,
    straight_through,
)

__all__ = [
    "Pooler",
    "KMeansPoolingConfig",
    "WardPoolingConfig",
    "KMeansPooler",
    "WardPooler",
    "POOLERS",
    "build_pooler",
    "Compressor",
    "build_compressor",
]

logger = logging.getLogger(__name__)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def _differentiable_pool(tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``tokens`` (m, H) by ``labels`` (m,), returning (k, H).

    ``labels`` carries no gradient (the assignment is treated as a constant); the
    average within each cluster is differentiable with respect to ``tokens``, so
    gradients flow back to every pooled token. Labels are densified so empty
    clusters never appear in the output.
    """
    _, inverse = torch.unique(labels, return_inverse=True)
    n_clusters = int(inverse.max().item()) + 1
    one_hot = F.one_hot(inverse, num_classes=n_clusters).to(tokens.dtype)
    counts = one_hot.sum(dim=0).clamp_min(1.0)
    return (one_hot.t() @ tokens) / counts.unsqueeze(1)


@dataclass(frozen=True)
class KMeansPoolingConfig:
    """Configuration for spherical k-means token pooling.

    The number of clusters is ``num_clusters`` when set, otherwise
    ``ceil_div(tail_len, pool_factor)`` -- a pooling *rate* rather than an absolute
    count. ``protected_tokens`` leading tokens (e.g. the ColBERT ``[D]`` marker)
    bypass pooling and are kept verbatim.
    """

    num_clusters: int | None = None
    pool_factor: int = 2
    protected_tokens: int = 1
    normalize_input: bool = True
    normalize_pooled: bool = True
    max_iter: int = 20
    seed: int = 13

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "KMeansPoolingConfig":
        data = data or {}
        num_clusters = data.get("num_clusters")
        return cls(
            num_clusters=None if num_clusters is None else int(num_clusters),
            pool_factor=int(data.get("pool_factor", 2)),
            protected_tokens=int(data.get("protected_tokens", 1)),
            normalize_input=_bool(data.get("normalize_input", True)),
            normalize_pooled=_bool(data.get("normalize_pooled", True)),
            max_iter=int(data.get("max_iter", 20)),
            seed=int(data.get("seed", 13)),
        )

    def __post_init__(self) -> None:
        if self.num_clusters is not None and self.num_clusters < 1:
            raise ValueError(f"num_clusters must be >= 1, got {self.num_clusters}.")
        if self.pool_factor < 1:
            raise ValueError(f"pool_factor must be >= 1, got {self.pool_factor}.")
        if self.protected_tokens < 0:
            raise ValueError(
                f"protected_tokens must be >= 0, got {self.protected_tokens}."
            )
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")


@dataclass(frozen=True)
class WardPoolingConfig:
    """Configuration for Ward hierarchical-agglomerative token pooling.

    Cut the dendrogram either by a target cluster count (``num_clusters`` or
    ``pool_factor``, ``criterion='maxclust'``) or by a cosine
    ``similarity_threshold`` in ``[-1, 1]`` (``criterion='distance'``): tokens
    closer than the threshold are merged, so the cluster count adapts to each
    document. Exactly the threshold path makes the output length data-dependent.
    """

    num_clusters: int | None = None
    pool_factor: int = 2
    similarity_threshold: float | None = None
    protected_tokens: int = 1
    normalize_input: bool = True
    normalize_pooled: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "WardPoolingConfig":
        data = data or {}
        num_clusters = data.get("num_clusters")
        threshold = data.get("similarity_threshold")
        return cls(
            num_clusters=None if num_clusters is None else int(num_clusters),
            pool_factor=int(data.get("pool_factor", 2)),
            similarity_threshold=None if threshold is None else float(threshold),
            protected_tokens=int(data.get("protected_tokens", 1)),
            normalize_input=_bool(data.get("normalize_input", True)),
            normalize_pooled=_bool(data.get("normalize_pooled", True)),
        )

    @property
    def criterion(self) -> str:
        return "distance" if self.similarity_threshold is not None else "maxclust"

    def __post_init__(self) -> None:
        if self.num_clusters is not None and self.num_clusters < 1:
            raise ValueError(f"num_clusters must be >= 1, got {self.num_clusters}.")
        if self.pool_factor < 1:
            raise ValueError(f"pool_factor must be >= 1, got {self.pool_factor}.")
        if self.protected_tokens < 0:
            raise ValueError(
                f"protected_tokens must be >= 0, got {self.protected_tokens}."
            )
        if self.similarity_threshold is not None:
            if not -1.0 <= self.similarity_threshold <= 1.0:
                raise ValueError(
                    "similarity_threshold must be a cosine similarity in [-1, 1], "
                    f"got {self.similarity_threshold}."
                )
            if not self.normalize_input:
                raise ValueError(
                    "similarity_threshold requires normalize_input=True so the cosine "
                    "threshold maps to a Euclidean cut on unit vectors."
                )


class Pooler(nn.Module):
    """Base class for differentiable token-pooling transforms.

    Subclasses implement :meth:`_cluster_labels` (the non-differentiable
    assignment, computed under ``no_grad``) and :meth:`get_config_dict`. The
    base class handles masking, protected tokens, the differentiable mean, and
    re-padding the variable-length pooled sequences back into a dense batch.
    """

    name: str = "pooler"

    def _target_clusters(self, tail_len: int) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def _cluster_labels(
        self, tokens: torch.Tensor, n_clusters: int
    ) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_config_dict(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def protected_tokens(self) -> int:
        return self.config.protected_tokens

    @property
    def normalize_pooled(self) -> bool:
        return self.config.normalize_pooled

    def _pool_one(self, valid: torch.Tensor) -> torch.Tensor:
        """Pool the valid (un-masked) tokens of a single sample, shape (n, H)."""
        protected_count = self.protected_tokens
        if valid.shape[0] <= protected_count + 1:
            return valid

        protected = valid[:protected_count]
        tail = valid[protected_count:]
        n_clusters = self._target_clusters(tail.shape[0])
        # A maxclust cut into >= the number of tokens is a no-op.
        if n_clusters >= tail.shape[0]:
            return valid

        with torch.no_grad():
            clustering_input = (
                _l2_normalize(tail) if self.config.normalize_input else tail
            )
            labels = self._cluster_labels(clustering_input, n_clusters)

        pooled = _differentiable_pool(tail, labels)
        if self.normalize_pooled:
            pooled = _l2_normalize(pooled)
        if protected_count == 0:
            return pooled
        return torch.cat([protected, pooled], dim=0)

    def forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool ``embeddings`` (B, T, H) given a boolean ``mask`` (B, T).

        Returns the pooled embeddings (B, K, H) and a fresh boolean mask (B, K),
        where ``K`` is the longest pooled sequence in the batch.
        """
        batch_size, _, hidden = embeddings.shape
        pooled_samples = [
            self._pool_one(embeddings[i][mask[i]]) for i in range(batch_size)
        ]
        max_len = max(sample.shape[0] for sample in pooled_samples)

        pooled = embeddings.new_zeros((batch_size, max_len, hidden))
        pooled_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.bool, device=embeddings.device
        )
        for i, sample in enumerate(pooled_samples):
            length = sample.shape[0]
            pooled[i, :length] = sample
            pooled_mask[i, :length] = True
        return pooled, pooled_mask


class KMeansPooler(Pooler):
    """Spherical k-means token pooling (GPU-friendly, differentiable mean)."""

    name = "kmeans"

    def __init__(self, config: KMeansPoolingConfig | None = None) -> None:
        super().__init__()
        self.config = config or KMeansPoolingConfig()

    def _target_clusters(self, tail_len: int) -> int:
        if self.config.num_clusters is not None:
            return min(self.config.num_clusters, tail_len)
        return max(-(-tail_len // self.config.pool_factor), 1)

    @torch.no_grad()
    def _cluster_labels(self, tokens: torch.Tensor, n_clusters: int) -> torch.Tensor:
        n_tokens = tokens.shape[0]
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        init = torch.randperm(n_tokens, generator=generator)[:n_clusters]
        centroids = _l2_normalize(tokens[init.to(tokens.device)])

        labels = tokens.new_zeros(n_tokens, dtype=torch.long)
        for _ in range(self.config.max_iter):
            similarities = tokens @ centroids.t()
            new_labels = similarities.argmax(dim=1)
            if torch.equal(new_labels, labels):
                break
            labels = new_labels
            one_hot = F.one_hot(labels, num_classes=n_clusters).to(tokens.dtype)
            counts = one_hot.sum(dim=0).clamp_min(1.0)
            centroids = _l2_normalize((one_hot.t() @ tokens) / counts.unsqueeze(1))
        return labels

    def get_config_dict(self) -> dict:
        return {"name": self.name, **self.config.__dict__}


class WardPooler(Pooler):
    """Ward hierarchical-agglomerative token pooling (cluster-count or threshold).

    Uses ``fastcluster`` + ``scipy`` for the linkage and dendrogram cut; the
    assignment runs on CPU under ``no_grad`` while the mean stays differentiable.
    """

    name = "ward"

    def __init__(self, config: WardPoolingConfig | None = None) -> None:
        super().__init__()
        self.config = config or WardPoolingConfig()

    def _target_clusters(self, tail_len: int) -> int:
        if self.config.criterion == "distance":
            # Decided per-document by the threshold cut, not a fixed count.
            return 1
        if self.config.num_clusters is not None:
            return min(self.config.num_clusters, tail_len)
        return max(-(-tail_len // self.config.pool_factor), 1)

    @torch.no_grad()
    def _cluster_labels(self, tokens: torch.Tensor, n_clusters: int) -> torch.Tensor:
        try:
            from scipy.cluster.hierarchy import fcluster
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "WardPooler requires `scipy` (optionally `fastcluster` for a faster "
                "linkage). Install it with `pip install scipy`."
            ) from exc
        try:
            # fastcluster is a faster drop-in for the Ward linkage when available.
            from fastcluster import linkage
        except ImportError:  # pragma: no cover - optional dependency
            from scipy.cluster.hierarchy import linkage

        points = tokens.detach().to(torch.float32).cpu().numpy()
        linkage_matrix = linkage(points, method="ward")
        if self.config.criterion == "distance":
            threshold = float(
                (max(0.0, 2.0 - 2.0 * self.config.similarity_threshold)) ** 0.5
            )
            criterion = "distance"
        else:
            threshold = float(n_clusters)
            criterion = "maxclust"
        labels = fcluster(linkage_matrix, t=threshold, criterion=criterion) - 1
        return torch.as_tensor(labels, dtype=torch.long, device=tokens.device)

    def get_config_dict(self) -> dict:
        return {"name": self.name, **self.config.__dict__}


# Registry of serializable poolers, keyed by their ``name``.
POOLERS: dict[str, type[Pooler]] = {
    cls.name: cls for cls in (KMeansPooler, WardPooler)
}

_POOLER_CONFIGS: dict[str, Any] = {
    "kmeans": KMeansPoolingConfig,
    "ward": WardPoolingConfig,
}


def build_pooler(spec: "Pooler | dict | None") -> Pooler | None:
    """Build a :class:`Pooler` from a friendly ``{"name": ..., **config}`` spec.

    Returns the instance unchanged if ``spec`` is already a :class:`Pooler`, and
    ``None`` for ``None`` / an empty spec / a spec with no ``name``. The same
    factory is used by the training loss and at index time, so a config atom maps
    to one pooler everywhere.
    """
    if spec is None:
        return None
    if isinstance(spec, Pooler):
        return spec
    spec = dict(spec)
    name = spec.pop("name", None)
    if name is None:
        return None
    pooler_cls = POOLERS.get(name)
    if pooler_cls is None:
        raise ValueError(f"Unknown pooler {name!r}. Available: {sorted(POOLERS)}.")
    config_cls = _POOLER_CONFIGS[name]
    return pooler_cls(config_cls.from_mapping(spec))


class Compressor(nn.Module):
    """Per-side (query or document) compression pipeline -- reused for training and inference.

    Applies an optional :class:`Pooler` (seq-len compression) followed by an
    optional :class:`~pylate.models.Quantizer` (value compression) to token
    embeddings. Either or both may be omitted; an empty pipeline is the identity
    (it only re-applies L2 normalization when ``pre_normalize`` is set). Pooling
    runs first so the quantizer sees the already-pooled vectors -- the same order
    used at indexing time (pool, then store quantized).

    The *same* object serves both phases, built once from one config atom:

    * **Training** -- :meth:`forward` runs on a padded ``(B, T, H)`` batch plus a
      boolean mask and wraps the quantizer in a straight-through estimator so
      gradients reach the full-precision weights. Used by
      :class:`~pylate.losses.CompressionAwareLoss`.
    * **Inference / indexing** -- :meth:`transform` runs on the per-document token
      tensors that :meth:`ColBERT.encode <pylate.models.ColBERT.encode>` returns
      (already skiplist/pad filtered, variable length, no grad) and yields the
      compressed vectors to store. Numerically identical to the compressed branch
      that training optimized.

    Parameters
    ----------
    quantizer
        Value-compression transform.
    pooler
        Seq-len-compression transform. Changes the token count (and, in training,
        the mask).
    pre_normalize
        L2-normalize the token embeddings before compressing. The loss feeds
        already-normalized embeddings, so this is idempotent there, but it keeps
        the pipeline correct when applied to raw ``encode`` output at index time.
    """

    def __init__(
        self,
        quantizer: Quantizer | None = None,
        pooler: Pooler | None = None,
        pre_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.quantizer = quantizer
        self.pooler = pooler
        self.pre_normalize = pre_normalize

    @property
    def is_identity(self) -> bool:
        return self.quantizer is None and self.pooler is None

    def forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training path: compress a padded ``(B, T, H)`` batch with a mask."""
        x = embeddings
        if self.pre_normalize:
            x = _l2_normalize(x)
        if self.pooler is not None:
            x, mask = self.pooler(x, mask)
        if self.quantizer is not None:
            x = straight_through(x, self.quantizer(x))
        return x, mask

    @torch.no_grad()
    def transform_one(self, embedding):
        """Inference path: compress one document's ``(n, H)`` token embeddings.

        Accepts (and returns) a torch tensor or a numpy array -- whatever
        :meth:`ColBERT.encode` produced. The tokens are assumed already filtered
        (skiplist + padding removed), so no mask is needed; the pooler's
        protected-token handling and the quantizer run exactly as in training.
        """
        was_tensor = torch.is_tensor(embedding)
        x = embedding if was_tensor else torch.as_tensor(embedding)
        x = x.float()
        if self.pre_normalize:
            x = _l2_normalize(x)
        if self.pooler is not None:
            x = self.pooler._pool_one(x)
        if self.quantizer is not None:
            x = self.quantizer(x)
        return x if was_tensor else x.cpu().numpy()

    def transform(self, embeddings):
        """Inference path: compress a list of per-document token tensors/arrays."""
        return [self.transform_one(embedding) for embedding in embeddings]

    def get_config_dict(self) -> dict:
        return {
            "pre_normalize": self.pre_normalize,
            "quantizer": _friendly_quantizer_spec(self.quantizer)
            if self.quantizer is not None
            else None,
            "pooler": self.pooler.get_config_dict()
            if self.pooler is not None
            else None,
        }

    @staticmethod
    def from_config(spec: dict | None) -> "Compressor":
        """Build a :class:`Compressor` from a ``{quantizer, pooler, pre_normalize}`` spec.

        Round-trips with :meth:`get_config_dict`, so a compressor saved after
        training rebuilds identically at index time.
        """
        spec = spec or {}
        return Compressor(
            quantizer=build_quantizer(spec.get("quantizer")),
            pooler=build_pooler(spec.get("pooler")),
            pre_normalize=_bool(spec.get("pre_normalize", True)),
        )


def build_compressor(spec: "Compressor | dict | None") -> Compressor:
    """Build a :class:`Compressor` from a config atom (or pass one through).

    The single entry point shared by the training runner and index-time code:
    ``{"quantizer": {...}, "pooler": {...}, "pre_normalize": bool}`` (any field
    optional). ``None`` / ``{}`` yields the identity compressor.
    """
    if isinstance(spec, Compressor):
        return spec
    return Compressor.from_config(spec)
