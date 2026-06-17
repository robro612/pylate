from __future__ import annotations

import json
import logging
import os
from typing import Callable

import torch
from torch import nn

__all__ = [
    "straight_through",
    "Quantizer",
    "IdentityQuantizer",
    "ScalarQuantizer",
    "BinaryQuantizer",
    "QUANTIZERS",
    "StraightThroughEstimator",
]

logger = logging.getLogger(__name__)


def straight_through(x: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator.

    Returns a tensor whose *value* is ``quantized`` (the forward pass sees the
    quantized embeddings) but whose *gradient* with respect to ``x`` is the
    identity (the backward pass behaves as if the quantization were absent).

    This is the classic ``x + (quantized - x).detach()`` trick. Because the
    detached residual carries no gradient, ``d output / d x == 1`` while
    ``output == quantized`` numerically. It works for *any* transform, even a
    non-differentiable one, which is what lets us train through an arbitrary
    quantizer with the unchanged loss functions.

    Parameters
    ----------
    x
        The full-precision embeddings (gradients flow back to these).
    quantized
        The output of applying an arbitrary transform to ``x``. Must be the
        same shape as ``x``.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(4, 8, requires_grad=True)
    >>> q = torch.round(x)  # non-differentiable on its own
    >>> y = straight_through(x, q)
    >>> bool(torch.equal(y, q))  # forward value is exactly the quantized one
    True
    >>> y.sum().backward()
    >>> bool(torch.equal(x.grad, torch.ones_like(x)))  # identity gradient
    True

    """
    return x + (quantized - x).detach()


class Quantizer(nn.Module):
    """Base class for (de)quantization transforms used for quantization-aware training.

    A quantizer maps full-precision embeddings to the values that would be
    obtained after quantizing and de-quantizing them, *without* any
    straight-through handling (that is applied by
    :class:`StraightThroughEstimator`). Subclasses implement :meth:`forward` and
    :meth:`get_config_dict` so that the configuration can be serialized.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_config_dict(self) -> dict:
        return {}


class IdentityQuantizer(Quantizer):
    """No-op quantizer. Useful as a baseline/sanity check for the STE wiring."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ScalarQuantizer(Quantizer):
    """Uniform affine scalar quantization to ``n_bits``, de-quantized back to float.

    The forward pass quantizes ``x`` to ``2 ** n_bits`` levels and immediately
    de-quantizes, so the output lives on the quantization grid while remaining a
    float tensor that the existing losses can consume.

    Parameters
    ----------
    n_bits
        Number of bits per value. ``8`` reproduces int8 storage, ``4`` int4, etc.
    symmetric
        If True, use a symmetric (zero-centered) grid spanning
        ``[-max(|x|), max(|x|)]``; otherwise an asymmetric grid spanning
        ``[min(x), max(x)]``. Symmetric matches signed-int storage; asymmetric
        matches the (u)int8 calibration used by
        ``sentence_transformers.quantization.quantize_embeddings``.
    value_range
        Optional fixed ``(min, max)`` range to quantize against. When ``None``
        (default) the range is computed dynamically per embedding vector (over
        the last dimension), which is parameter-free and well-suited to QAT.

    Examples
    --------
    >>> import torch
    >>> q = ScalarQuantizer(n_bits=8)
    >>> x = torch.randn(2, 16)
    >>> q(x).shape
    torch.Size([2, 16])

    """

    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        value_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        if n_bits < 1:
            raise ValueError(f"n_bits must be >= 1, got {n_bits}.")
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.value_range = tuple(value_range) if value_range is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(x.dtype).eps

        if self.symmetric:
            levels = 2 ** (self.n_bits - 1) - 1
            if self.value_range is not None:
                bound = max(abs(self.value_range[0]), abs(self.value_range[1]))
                scale = torch.full_like(x[..., :1], max(bound, eps)) / levels
            else:
                scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / levels
            q = torch.round(x / scale).clamp(-levels, levels)
            return q * scale

        levels = 2**self.n_bits - 1
        if self.value_range is not None:
            xmin = torch.full_like(x[..., :1], float(self.value_range[0]))
            xmax = torch.full_like(x[..., :1], float(self.value_range[1]))
        else:
            xmin = x.amin(dim=-1, keepdim=True)
            xmax = x.amax(dim=-1, keepdim=True)
        scale = (xmax - xmin).clamp_min(eps) / levels
        q = torch.round((x - xmin) / scale).clamp(0, levels)
        return q * scale + xmin

    def get_config_dict(self) -> dict:
        return {
            "n_bits": self.n_bits,
            "symmetric": self.symmetric,
            "value_range": list(self.value_range)
            if self.value_range is not None
            else None,
        }


class BinaryQuantizer(Quantizer):
    """Binarizes embeddings to ``{-scale, +scale}`` via the sign function.

    Parameters
    ----------
    scale
        The magnitude assigned to each binarized value. ``1.0`` yields a pure
        sign; values are otherwise mapped to ``+/- scale``.

    """

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sign(0) == 0; map zeros to +scale so the output is strictly binary.
        signs = torch.sign(x)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return signs * self.scale

    def get_config_dict(self) -> dict:
        return {"scale": self.scale}


# Registry of serializable quantizers, keyed by class name. Custom quantizers
# can be registered here (or simply passed as a callable, in which case they
# cannot be serialized -- see StraightThroughEstimator).
QUANTIZERS: dict[str, type[Quantizer]] = {
    cls.__name__: cls for cls in (IdentityQuantizer, ScalarQuantizer, BinaryQuantizer)
}


def _build_quantizer(spec: dict | None) -> Quantizer:
    """Reconstruct a :class:`Quantizer` from a serialized ``{type, config}`` spec."""
    if spec is None or spec.get("quantizer_type") is None:
        return IdentityQuantizer()
    quantizer_type = spec["quantizer_type"]
    quantizer_cls = QUANTIZERS.get(quantizer_type)
    if quantizer_cls is None:
        logger.warning(
            f"Unknown quantizer '{quantizer_type}'; falling back to IdentityQuantizer. "
            "Register it in pylate.models.QUANTIZERS to restore it."
        )
        return IdentityQuantizer()
    return quantizer_cls(**(spec.get("quantizer_config") or {}))


def _quantizer_spec(transform) -> dict | None:
    """Serialize a transform to a ``{type, config}`` spec, or ``None`` if it cannot be."""
    if isinstance(transform, Quantizer):
        return {
            "quantizer_type": type(transform).__name__,
            "quantizer_config": transform.get_config_dict(),
        }
    # Arbitrary callable / non-Quantizer module: not serializable.
    return None


class StraightThroughEstimator(nn.Module):
    """Quantization-aware training module driven by a straight-through estimator.

    Append this to a :class:`~pylate.models.ColBERT` model (after the ``Dense``
    projection) to make every embedding the loss sees pass through an arbitrary
    ``transform`` (typically a quantizer). On the forward pass the embeddings are
    quantized; on the backward pass gradients flow straight through to the
    full-precision weights (see :func:`straight_through`). Training therefore
    uses the *same* loss functions while optimizing the network to be robust to
    the chosen quantization.

    The module operates on ``features["token_embeddings"]`` -- exactly the field
    the losses read -- so no loss code needs to change. It is a permanent module
    of the model: it is saved/loaded with the checkpoint and is active at
    inference (``encode``) as well, so the model emits embeddings on the
    quantization grid everywhere.

    **Asymmetric quantization.** Queries and documents can use different
    transforms (e.g. int8 queries, binary documents). The module reads the
    ``is_query`` flag that :meth:`ColBERT.tokenize <pylate.models.ColBERT.tokenize>`
    injects into ``features`` to route each input to the right transform. Pass
    ``query_transform`` and ``document_transform`` for the asymmetric case, or a
    single ``transform`` for the symmetric case.

    Parameters
    ----------
    transform
        The transform used for both queries and documents (symmetric case).
        Either a :class:`Quantizer` instance (serializes with the model) or any
        callable ``Tensor -> Tensor`` of the same shape (works for training but
        is not serialized; the model loads it back as a no-op and warns).
        Defaults to :class:`IdentityQuantizer`.
    query_transform
        Transform applied to queries. If set, enables asymmetric mode. Falls back
        to ``transform`` (or identity) when omitted.
    document_transform
        Transform applied to documents. If set, enables asymmetric mode. Falls
        back to ``transform`` (or identity) when omitted.
    pre_normalize
        If True (default), L2-normalize the embeddings over the last dimension
        before applying the transform. This matches inference, where ColBERT
        stores L2-normalized embeddings and quantizes those, so the quantizer
        sees the same value distribution it will see at serving time. The losses
        re-normalize afterwards, which is a no-op direction-wise.
    default_is_query
        Which transform to use when ``features`` carries no ``is_query`` flag
        (e.g. when calling the module on a hand-built features dict). Defaults to
        ``False`` (document transform).

    Examples
    --------
    >>> import torch
    >>> from pylate.models import (
    ...     StraightThroughEstimator,
    ...     ScalarQuantizer,
    ...     BinaryQuantizer,
    ... )
    >>> _ = torch.manual_seed(0)
    >>> # Asymmetric: int8 queries, binary documents.
    >>> ste = StraightThroughEstimator(
    ...     query_transform=ScalarQuantizer(n_bits=8),
    ...     document_transform=BinaryQuantizer(),
    ... )
    >>> q = ste({"token_embeddings": torch.randn(3, 32), "is_query": True})
    >>> d = ste({"token_embeddings": torch.randn(3, 32), "is_query": False})
    >>> bool((d["token_embeddings"].abs() == 1.0).all())  # documents are binary
    True
    >>> int(q["token_embeddings"].unique().numel()) > 2  # queries are int8 (many levels)
    True

    """

    def __init__(
        self,
        transform: Quantizer | Callable[[torch.Tensor], torch.Tensor] | None = None,
        query_transform: Quantizer | Callable[[torch.Tensor], torch.Tensor] | None = None,
        document_transform: Quantizer
        | Callable[[torch.Tensor], torch.Tensor]
        | None = None,
        pre_normalize: bool = True,
        default_is_query: bool = False,
    ) -> None:
        super().__init__()

        self.asymmetric = query_transform is not None or document_transform is not None
        fallback = transform if transform is not None else IdentityQuantizer()
        if self.asymmetric:
            query = query_transform if query_transform is not None else fallback
            document = document_transform if document_transform is not None else fallback
        else:
            # Share a single transform object for both routes.
            query = document = fallback

        # Register Quantizer (nn.Module) transforms so .to()/.train()/.eval() and
        # any parameters they hold propagate. Plain callables are kept as bare
        # attributes. We store the two routes under distinct names; when they are
        # the same shared object, registering it twice is harmless (quantizers
        # are parameter-free by default).
        self._set_route("query", query)
        self._set_route("document", document)

        self.pre_normalize = pre_normalize
        self.default_is_query = default_is_query

    def _set_route(self, name: str, transform) -> None:
        if isinstance(transform, nn.Module):
            setattr(self, f"{name}_transform", transform)
            setattr(self, f"_{name}_fn", None)
        else:
            # nn.Module.__setattr__ would reject a plain callable as a submodule
            # attribute name clash, so keep the module attribute None.
            setattr(self, f"{name}_transform", None)
            object.__setattr__(self, f"_{name}_fn", transform)

    def _transform_for(self, is_query: bool):
        name = "query" if is_query else "document"
        module = getattr(self, f"{name}_transform")
        return module if module is not None else getattr(self, f"_{name}_fn")

    @staticmethod
    def _resolve_is_query(features: dict, default: bool) -> bool:
        flag = features.get("is_query", None)
        if flag is None:
            return default
        if torch.is_tensor(flag):
            # Collated as a single per-column scalar; take any element.
            return bool(flag.reshape(-1)[0].item())
        return bool(flag)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = features["token_embeddings"]
        if self.pre_normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
        is_query = self._resolve_is_query(features, self.default_is_query)
        transform = self._transform_for(is_query)
        quantized = transform(x)
        features["token_embeddings"] = straight_through(x, quantized)
        return features

    def get_config_dict(self) -> dict:
        return {
            "asymmetric": self.asymmetric,
            "pre_normalize": self.pre_normalize,
            "default_is_query": self.default_is_query,
            "query": _quantizer_spec(self._transform_for(is_query=True)),
            "document": _quantizer_spec(self._transform_for(is_query=False)),
        }

    def save(self, output_path: str, *args, **kwargs) -> None:
        os.makedirs(output_path, exist_ok=True)
        config = self.get_config_dict()
        for route in ("query", "document"):
            if config[route] is None:
                logger.warning(
                    "StraightThroughEstimator %s transform is non-serializable (a "
                    "plain callable or non-Quantizer module); it will be loaded back "
                    "as a no-op IdentityQuantizer. Use a Quantizer subclass (e.g. "
                    "ScalarQuantizer) to persist the configuration.",
                    route,
                )
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def load(input_path) -> "StraightThroughEstimator":
        with open(os.path.join(input_path, "config.json")) as f:
            config = json.load(f)
        if config.get("asymmetric", False):
            return StraightThroughEstimator(
                query_transform=_build_quantizer(config.get("query")),
                document_transform=_build_quantizer(config.get("document")),
                pre_normalize=config.get("pre_normalize", True),
                default_is_query=config.get("default_is_query", False),
            )
        return StraightThroughEstimator(
            transform=_build_quantizer(config.get("query")),
            pre_normalize=config.get("pre_normalize", True),
            default_is_query=config.get("default_is_query", False),
        )
