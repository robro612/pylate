"""RLHN hard-negative datasets, adapted to PyLate contrastive training.

`RLHN <https://huggingface.co/rlhn>`_ ships retriever training data as
``query`` + ``positive_passages`` (a list) + ``negative_passages`` (mined hard
negatives, each ``{docid, text, title}``), with no teacher scores -- i.e. it is
contrastive-with-hard-negatives data. :func:`rlhn_to_contrastive` flattens a row
into the ``query`` / ``positive`` / ``negative_1..k`` columns that
:class:`~pylate.losses.Contrastive` expects (the query column stays named
``query`` so the collator routes it as the query side). The hard negatives join
the in-batch negatives during scoring.
"""

from __future__ import annotations

import random
from typing import Any

__all__ = ["passage_text", "rlhn_to_contrastive", "load_rlhn"]


def passage_text(passage: dict) -> str:
    """Render an RLHN ``{docid, text, title}`` passage as a single string."""
    title = (passage.get("title") or "").strip()
    text = (passage.get("text") or "").strip()
    return f"{title}. {text}" if title else text


def rlhn_to_contrastive(dataset: Any, num_negatives: int = 7, seed: int = 42) -> Any:
    """Flatten an RLHN-style ``Dataset`` to contrastive ``query/positive/negative_*`` columns.

    Parameters
    ----------
    dataset
        A Hugging Face ``Dataset`` with ``query``, ``positive_passages`` and
        ``negative_passages`` columns.
    num_negatives
        Number of hard negatives to emit per query (``negative_1 .. negative_k``).
        Rows with fewer mined negatives are topped up by resampling with
        replacement so every row has the same columns.
    seed
        Base seed for the per-row negative sampling (deterministic).
    """
    dataset = dataset.filter(
        lambda row: bool(row["positive_passages"]) and bool(row["negative_passages"])
    )

    def convert(row: dict, index: int) -> dict:
        rng = random.Random(seed + index)
        negatives = row["negative_passages"]
        if len(negatives) >= num_negatives:
            chosen = rng.sample(negatives, num_negatives)
        else:
            chosen = list(negatives) + [
                rng.choice(negatives) for _ in range(num_negatives - len(negatives))
            ]
        converted = {
            "query": row["query"],
            "positive": passage_text(row["positive_passages"][0]),
        }
        for i in range(num_negatives):
            converted[f"negative_{i + 1}"] = passage_text(chosen[i])
        return converted

    return dataset.map(
        convert, with_indices=True, remove_columns=dataset.column_names
    )


def load_rlhn(
    name: str = "rlhn/rlhn-100K",
    *,
    num_negatives: int = 7,
    seed: int = 42,
    split: str = "train",
    max_samples: int | None = None,
) -> Any:
    """Load an RLHN dataset and flatten it for contrastive training.

    Examples
    --------
    >>> from pylate import data  # doctest: +SKIP
    >>> train = data.load_rlhn("rlhn/rlhn-100K", num_negatives=7)  # doctest: +SKIP
    """
    from datasets import load_dataset

    dataset = load_dataset(name, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return rlhn_to_contrastive(dataset, num_negatives=num_negatives, seed=seed)
