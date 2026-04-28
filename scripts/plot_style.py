"""Canonical plot styles for paper figures.

Provides consistent color + linestyle per model name, shared across
score_distributions.py, token_retrieval_analysis.py, and other scripts.

Design:
    ColBERT baselines: blue (#2E86AB), solid line
    XTR variants: Oranges_r gradient (dark->light by k), distinct linestyles
    XTR multi-k: red (#d62728)
    Same k gets same color+linestyle across KD/contrastive (separate plots).
"""

COLBERT_COLOR = "#2E86AB"

_XTR_COLORS = {
    "k64": "#a63603",
    "k128": "#d94801",
    "k256": "#f16913",
    "k512": "#fd8e3d",
    "multik": "#c4267d",
}

MODEL_STYLE = {
    # External baselines
    "colbert": {"color": COLBERT_COLOR, "linestyle": "-"},
    "google_xtr": {"color": _XTR_COLORS["k128"], "linestyle": "-"},
    "gte_moderncolbert_v1": {"color": "#6C757D", "linestyle": "-"},
    # ColBERT-trained (one per subplot, always solid)
    "mb_colbert_kd": {"color": COLBERT_COLOR, "linestyle": "-"},
    "mb_colbert_contrastive": {"color": COLBERT_COLOR, "linestyle": "-"},
    # XTR-KD
    "mb_xtr_kd_k64": {"color": _XTR_COLORS["k64"], "linestyle": "--"},
    "mb_xtr_kd_k128": {"color": _XTR_COLORS["k128"], "linestyle": "-."},
    "mb_xtr_kd_k256": {"color": _XTR_COLORS["k256"], "linestyle": ":"},
    "mb_xtr_kd_k512": {"color": _XTR_COLORS["k512"], "linestyle": (0, (3, 1, 1, 1))},
    "mb_xtr_kd_multik": {"color": _XTR_COLORS["multik"], "linestyle": "--"},
    # XTR-C (same color/linestyle per k)
    "mb_xtr_contrastive_k64": {"color": _XTR_COLORS["k64"], "linestyle": "--"},
    "mb_xtr_contrastive_k128": {"color": _XTR_COLORS["k128"], "linestyle": "-."},
    "mb_xtr_contrastive_k256": {"color": _XTR_COLORS["k256"], "linestyle": ":"},
    "mb_xtr_contrastive_k512": {"color": _XTR_COLORS["k512"], "linestyle": (0, (3, 1, 1, 1))},
    "mb_xtr_contrastive_multik": {"color": _XTR_COLORS["multik"], "linestyle": "--"},
}


def get_style(model_name):
    """Return plot kwargs (color, linestyle) for a model name.

    Returns an empty dict for unknown models, letting matplotlib pick defaults.
    """
    return dict(MODEL_STYLE.get(model_name, {}))
