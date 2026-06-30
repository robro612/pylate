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
    "CastQuantizer",
    "SHBQQuantizer",
    "build_quantizer",
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


class CastQuantizer(Quantizer):
    """Quantize by casting to an explicit output ``dtype``, then back to float.

    The output dtype *is* the knob -- this models "store the embedding in this
    format," which is what actually happens at serving time:

    * **Float formats** (``float32`` / ``float16`` / ``bfloat16``) use a true IEEE
      cast (``x.to(dtype).to(x.dtype)``). These are *non-uniform* (denser near
      zero), so they are not the same as uniform N-bit quantization; ``float32``
      is a no-op. They carry so much resolution that they are effectively no-ops
      for QAT, but are available for completeness / inference parity.
    * **Integer formats** (``int8`` / ``uint8``) use uniform affine quantization to
      that integer's range, de-quantized back to float so the value lives on the
      int grid. ``int8`` is symmetric (per-vector max-abs, 127 levels), matching
      the scrambled-Hadamard int8 query path; ``uint8`` is asymmetric over
      ``[min, max]`` (255 levels), matching ``quantize_embeddings``.

    Parameters
    ----------
    dtype
        Output format: ``"float32"``, ``"float16"`` (``"fp16"``), ``"bfloat16"``
        (``"bf16"``), ``"int8"``, or ``"uint8"``.
    value_range
        Optional fixed ``(min, max)`` calibration range for the integer formats.
        ``None`` (default) calibrates dynamically per embedding vector.

    Examples
    --------
    >>> import torch
    >>> CastQuantizer("int8")(torch.randn(2, 8)).shape
    torch.Size([2, 8])
    >>> x = torch.randn(2, 8)
    >>> bool(torch.equal(CastQuantizer("float32")(x), x))  # fp32 is a no-op
    True

    """

    _FLOAT_DTYPES = {
        "float32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    _INT_DTYPES = {"int8", "uint8"}

    def __init__(
        self,
        dtype: str = "int8",
        value_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        key = str(dtype).lower().replace("torch.", "")
        if key not in self._FLOAT_DTYPES and key not in self._INT_DTYPES:
            valid = sorted({*self._FLOAT_DTYPES, *self._INT_DTYPES})
            raise ValueError(f"Unknown dtype {dtype!r}. Expected one of {valid}.")
        # Canonicalize aliases so the saved config is unambiguous.
        if key in self._FLOAT_DTYPES:
            self.dtype = {
                torch.float32: "float32",
                torch.float16: "float16",
                torch.bfloat16: "bfloat16",
            }[self._FLOAT_DTYPES[key]]
        else:
            self.dtype = key
        self.value_range = tuple(value_range) if value_range is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dtype in self._FLOAT_DTYPES:
            return x.to(self._FLOAT_DTYPES[self.dtype]).to(x.dtype)

        eps = torch.finfo(x.dtype).eps
        if self.dtype == "int8":
            levels = 127
            if self.value_range is not None:
                bound = max(abs(self.value_range[0]), abs(self.value_range[1]))
                scale = torch.full_like(x[..., :1], max(bound, eps)) / levels
            else:
                scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / levels
            return torch.round(x / scale).clamp(-levels, levels) * scale

        # uint8: asymmetric affine over [min, max].
        levels = 255
        if self.value_range is not None:
            xmin = torch.full_like(x[..., :1], float(self.value_range[0]))
            xmax = torch.full_like(x[..., :1], float(self.value_range[1]))
        else:
            xmin = x.amin(dim=-1, keepdim=True)
            xmax = x.amax(dim=-1, keepdim=True)
        scale = (xmax - xmin).clamp_min(eps) / levels
        return torch.round((x - xmin) / scale).clamp(0, levels) * scale + xmin

    def get_config_dict(self) -> dict:
        return {
            "dtype": self.dtype,
            "value_range": list(self.value_range)
            if self.value_range is not None
            else None,
        }


def _build_hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Normalized Sylvester--Hadamard matrix of size ``dim`` (power of two).

    Divided by ``sqrt(dim)`` so the matrix is orthonormal (``H @ H == I`` since the
    Sylvester construction is symmetric). This makes the scrambled-Hadamard
    transform a norm-preserving rotation, so dot products -- and therefore MaxSim
    -- are invariant to it.
    """
    if dim < 1 or (dim & (dim - 1)):
        raise ValueError(f"SHBQ dimension must be a power of two, got {dim}.")
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < dim:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h / (dim**0.5)


class SHBQQuantizer(Quantizer):
    """Scrambled-Hadamard quantization: an orthonormal rotation around an inner quantizer.

    Applies a fixed random sign flip followed by a (normalized) Hadamard rotation
    -- ``r = (x * d) @ H`` -- runs an ``inner`` quantizer **in that rotated space**,
    then rotates the result back into the original space (``(q @ H) * d``). The
    rotation is the entire contribution of SHBQ: because it is orthonormal it
    spreads each coordinate's information evenly, so quantizing the rotated vector
    loses far less than quantizing the raw one. The de-quantized output stays in
    the input's coordinate system, so :func:`straight_through` is well defined and
    the existing MaxSim losses consume it unchanged.

    The inner quantizer is what decides the *format* in the rotated space:

    * ``BinaryQuantizer()`` (default) -- canonical SHBQ binary documents.
    * ``CastQuantizer("int8")`` -- the scrambled-Hadamard int8 query companion.

    This is exactly ``rotation(inner(rotation^{-1}))``; with ``BinaryQuantizer`` the
    formulation is identical to a plain sign on the rotated vector.

    Queries and documents must share the same ``(H, d)`` for their scores to match
    the rotated-space dot product. Since ``H`` is determined by ``dim`` and ``d``
    by ``seed``, two SHBQ quantizers with the same ``dim`` and ``seed`` are
    automatically consistent (e.g. an int8-query / binary-document pair).

    Parameters
    ----------
    inner
        Quantizer applied in the rotated space. A :class:`Quantizer` instance, a
        ``{"type": ..., **config}`` spec, or ``None`` (defaults to
        :class:`BinaryQuantizer`).
    dim
        Embedding dimension (a power of two). If ``None`` it is inferred from the
        first input and the transform is built lazily.
    seed
        Seed for the Rademacher sign vector ``d``. Must match across the query and
        document quantizers for asymmetric setups.

    Examples
    --------
    >>> import torch
    >>> _ = torch.manual_seed(0)
    >>> q = SHBQQuantizer(dim=128)  # binary documents
    >>> q(torch.randn(4, 128)).shape
    torch.Size([4, 128])
    >>> # int8 queries score against binary documents (shared dim+seed).
    >>> query = SHBQQuantizer(inner=CastQuantizer("int8"), dim=128)
    >>> doc = SHBQQuantizer(inner=BinaryQuantizer(), dim=128)
    >>> (query(torch.randn(3, 128)) @ doc(torch.randn(5, 128)).T).shape
    torch.Size([3, 5])

    """

    def __init__(
        self,
        inner: "Quantizer | dict | None" = None,
        dim: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if inner is None:
            inner = BinaryQuantizer()
        elif isinstance(inner, dict):
            inner = build_quantizer(inner)
        self.inner = inner
        self.seed = seed
        self.dim = dim
        # Registered (initially empty) so they move with .to()/.cuda() and load
        # from the state dict; populated by _ensure_transform on first use.
        self.register_buffer("H", None)
        self.register_buffer("d", None)
        if dim is not None:
            self._ensure_transform(dim, torch.float32, torch.device("cpu"))

    def _ensure_transform(
        self, dim: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        if self.H is not None and self.H.shape[0] == dim:
            return
        if self.dim is not None and dim != self.dim:
            raise ValueError(
                f"SHBQQuantizer was configured for dim={self.dim} but received "
                f"input of dim={dim}."
            )
        self.H = _build_hadamard(dim, dtype=dtype, device=device)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        signs = torch.randint(0, 2, (dim,), generator=generator).to(dtype) * 2 - 1
        self.d = signs.to(device=device)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_transform(x.shape[-1], x.dtype, x.device)
        h = self.H.to(dtype=x.dtype, device=x.device)
        d = self.d.to(dtype=x.dtype, device=x.device)

        rotated = (x * d) @ h
        quantized = self.inner(rotated)
        # Rotate back into the original coordinate system (H is orthonormal and
        # symmetric, d is its own inverse), so the output is same-space as x.
        return (quantized @ h) * d

    def get_config_dict(self) -> dict:
        return {
            "inner": _friendly_quantizer_spec(self.inner),
            "dim": self.dim,
            "seed": self.seed,
        }


# Registry of serializable quantizers, keyed by class name. Custom quantizers
# can be registered here (or simply passed as a callable, in which case they
# cannot be serialized -- see StraightThroughEstimator).
QUANTIZERS: dict[str, type[Quantizer]] = {
    cls.__name__: cls
    for cls in (
        IdentityQuantizer,
        ScalarQuantizer,
        BinaryQuantizer,
        CastQuantizer,
        SHBQQuantizer,
    )
}


def build_quantizer(spec: "Quantizer | dict | None") -> Quantizer | None:
    """Build a :class:`Quantizer` from a friendly ``{"type": ..., **config}`` spec.

    Returns the instance unchanged if ``spec`` is already a :class:`Quantizer`, and
    ``None`` for ``None`` / an empty spec / a spec with no ``type``. Nested specs
    (e.g. :class:`SHBQQuantizer`'s ``inner``) are resolved recursively by the
    target class. Unknown types raise, so config typos fail loudly.
    """
    if spec is None:
        return None
    if isinstance(spec, Quantizer):
        return spec
    spec = dict(spec)
    quantizer_type = spec.pop("type", None)
    if quantizer_type is None:
        return None
    if quantizer_type not in QUANTIZERS:
        raise ValueError(
            f"Unknown quantizer {quantizer_type!r}. Available: {sorted(QUANTIZERS)}."
        )
    return QUANTIZERS[quantizer_type](**spec)


def _friendly_quantizer_spec(transform) -> dict | None:
    """Serialize a quantizer to a friendly ``{"type": ..., **config}`` spec."""
    if isinstance(transform, Quantizer):
        return {"type": type(transform).__name__, **transform.get_config_dict()}
    return None


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
