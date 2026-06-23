"""Load ViDoRe visual-document-retrieval datasets in the BEIR triple shape used
by the retrieval benchmark (``documents``, ``queries``, ``qrels``).

Unlike :class:`~pylate.evaluation.vidore_evaluator.ViDoREvaluator` (which runs an
exact brute-force scoring pass), this loader returns the raw corpus/queries/qrels
so the documents (page **images**) can be encoded, cached, indexed (PLAID / PGC),
and retrieved through the same pipeline as text BEIR datasets.

Documents carry an ``"image"`` key (a PIL image) instead of ``"text"`` — the rest
of the pipeline is modality-agnostic because it operates on multi-vector
embeddings, not raw inputs. See ``scripts/benchmark_indexes.py`` for the wiring.
"""

from __future__ import annotations

import logging

from .vidore_evaluator import ALL_VIDORE_DATASETS

logger = logging.getLogger(__name__)


def resolve_vidore_repo(dataset_name: str) -> str:
    """Resolve a short ViDoRe name (e.g. ``"finance"``) or a full ``vidore/*``
    HF repo id to the canonical HF repo id."""
    key = dataset_name.lower()
    if key in ALL_VIDORE_DATASETS:
        return ALL_VIDORE_DATASETS[key]
    # Already a full repo id (e.g. "vidore/vidore_v3_finance_en").
    return dataset_name


def load_vidore(
    dataset_name: str,
    split: str = "test",
) -> tuple[list[dict], dict[str, str], dict[str, dict[str, int]]]:
    """Load a ViDoRe BEIR-format dataset.

    Parameters
    ----------
    dataset_name
        A short name from :data:`ALL_VIDORE_DATASETS` (``"finance"``, ``"hr"``…)
        or a full ``vidore/*`` HF repo id.
    split
        Dataset split to load. ViDoRe ships everything under ``"test"``.

    Returns
    -------
    (documents, queries, qrels)
        ``documents``: list of ``{"id": str, "image": PIL.Image}`` in corpus order.
        ``queries``: ``{query_id: query_text}``.
        ``qrels``: ``{query_id: {corpus_id: relevance}}`` (graded relevance kept).
    """
    from datasets import load_dataset

    repo = resolve_vidore_repo(dataset_name)
    logger.info("Loading ViDoRe dataset %s (repo=%s)", dataset_name, repo)

    corpus = load_dataset(repo, "corpus", split=split)
    query_rows = load_dataset(repo, "queries", split=split)
    qrel_rows = load_dataset(repo, "qrels", split=split)

    documents = [
        {"id": str(row["corpus_id"]), "image": row["image"].convert("RGB")}
        for row in corpus
    ]

    queries = {str(row["query_id"]): row["query"].strip() for row in query_rows}

    qrels: dict[str, dict[str, int]] = {}
    for row in qrel_rows:
        qid = str(row["query_id"])
        qrels.setdefault(qid, {})[str(row["corpus_id"])] = int(row["score"])

    logger.info(
        "Loaded %d images, %d queries, %d queries with qrels",
        len(documents),
        len(queries),
        len(qrels),
    )
    return documents, queries, qrels
