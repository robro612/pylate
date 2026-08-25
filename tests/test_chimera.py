"""Test suite for the Chimera index backend.

Chimera is a compiled C++/CUDA extension installed only in the cu130
environments (scripts/sync_env.sh builds the `chimera` extra there), and every
code path here needs a GPU, so the whole module skips when either is missing.
"""

import shutil
import uuid

import numpy as np
import pytest
import torch

from pylate import indexes, retrieve

if not torch.cuda.is_available():
    pytest.skip("Chimera requires a CUDA device", allow_module_level=True)

try:
    from pylate.indexes.chimera import _load_chimera

    _load_chimera()
except ImportError as exc:  # pragma: no cover - depends on local build state
    pytest.skip(f"Chimera extension not built: {exc}", allow_module_level=True)


_chimera = _load_chimera()
DIM = 128
# Whatever this environment was compiled for; sync_env.sh builds 48 so the
# 48-token BEIR datasets work, but the suite must not assume that.
QUERY_TOKENS = getattr(_chimera, "Q_DOCLEN", 32)
N_DOCUMENTS = 400


@pytest.fixture(scope="module")
def corpus():
    """Random unit-norm token embeddings, one variable-length document each.

    Using a document's own tokens as the query gives a ground truth that does
    not depend on a model: MaxSim of a document against itself is its token
    count, an order of magnitude above any other document here, so the right
    answer at rank 1 is unambiguous.
    """
    rng = np.random.default_rng(0)
    doclens = rng.integers(QUERY_TOKENS + 8, QUERY_TOKENS + 58, size=N_DOCUMENTS)
    documents = []
    for length in doclens:
        vectors = rng.standard_normal((length, DIM)).astype(np.float32)
        documents.append(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
    ids = [f"doc_{i}" for i in range(N_DOCUMENTS)]
    return ids, documents


@pytest.fixture(scope="module")
def index(corpus, tmp_path_factory):
    ids, documents = corpus
    folder = tmp_path_factory.mktemp("chimera")
    index = indexes.Chimera(
        index_folder=str(folder),
        index_name=f"test_{uuid.uuid4().hex[:8]}",
        override=True,
        n_clusters=256,
        ex_bits=4,
        nprobe=64,
        k_refine=400,
        k_full_bit=200,
    )
    index.add_documents(documents_ids=ids, documents_embeddings=documents)
    yield index
    shutil.rmtree(folder, ignore_errors=True)


def test_is_end_to_end(index):
    assert index.is_end_to_end_index is True


def test_build_metadata(index, corpus):
    _, documents = corpus
    assert index.is_indexed
    assert index.actual_total_centroids == 256
    assert index.dim == DIM
    assert index.n_tokens == sum(len(d) for d in documents)


def test_self_query_ranks_its_own_document_first(index, corpus):
    ids, documents = corpus
    targets = [0, 17, 200, 399]
    queries = [documents[t][:QUERY_TOKENS] for t in targets]
    results = index(queries, k=10)
    assert [r[0]["id"] for r in results] == [ids[t] for t in targets]


def test_results_are_ranked_by_descending_score(index, corpus):
    _, documents = corpus
    results = index([documents[5][:QUERY_TOKENS]], k=10)
    scores = [r["score"] for r in results[0]]
    assert scores == sorted(scores, reverse=True)


def test_search_is_deterministic(index, corpus):
    """The index PyLate searches must be the reloaded one.

    Searching the object ``ChimeraIndex.build`` returns is unstable upstream —
    it drops the correct top-1 roughly one call in ten — so ``_build`` discards
    it and reloads from disk. This is the regression test for that.
    """
    _, documents = corpus
    query = [documents[42][:QUERY_TOKENS]]
    runs = {tuple(r["id"] for r in index(query, k=10)[0]) for _ in range(25)}
    assert len(runs) == 1


def test_short_queries_are_zero_padded(index, corpus):
    """Padding to Q_DOCLEN is score-preserving: a zero vector scores 0 anywhere."""
    ids, documents = corpus
    short = documents[77][: QUERY_TOKENS - 10]
    assert index([short], k=5)[0][0]["id"] == ids[77]


def test_long_queries_raise_rather_than_truncate(index):
    rng = np.random.default_rng(1)
    too_long = rng.standard_normal((QUERY_TOKENS + 16, DIM)).astype(np.float32)
    with pytest.raises(ValueError, match="Q_DOCLEN"):
        index([too_long], k=5)


def test_query_dimension_mismatch_raises(index):
    rng = np.random.default_rng(2)
    wrong_dim = rng.standard_normal((QUERY_TOKENS, 64)).astype(np.float32)
    with pytest.raises(ValueError, match="dimension"):
        index([wrong_dim], k=5)


def test_torch_queries_are_accepted(index, corpus):
    ids, documents = corpus
    query = torch.from_numpy(documents[9][:QUERY_TOKENS])
    assert index([query], k=5)[0][0]["id"] == ids[9]


def test_reload_from_disk_preserves_results(index, corpus):
    ids, documents = corpus
    query = [documents[123][:QUERY_TOKENS]]
    before = [r["id"] for r in index(query, k=10)[0]]

    reloaded = indexes.Chimera(
        index_folder=index.index_folder,
        index_name=index.index_name,
        override=False,
        nprobe=index.nprobe,
        k_refine=index.k_refine,
        k_full_bit=index.k_full_bit,
    )
    assert reloaded.actual_total_centroids == index.actual_total_centroids
    assert [r["id"] for r in reloaded(query, k=10)[0]] == before


def test_retriever_integration(index, corpus):
    ids, documents = corpus
    retriever = retrieve.ColBERT(index=index)
    results = retriever.retrieve(
        queries_embeddings=[documents[3][:QUERY_TOKENS]], k=5
    )
    assert results[0][0]["id"] == ids[3]


def test_nprobe_above_the_cagra_limit_raises(tmp_path):
    with pytest.raises(ValueError, match="1024"):
        indexes.Chimera(
            index_folder=str(tmp_path), index_name="bad_nprobe", nprobe=2048
        )


def test_k_full_bit_above_k_refine_raises(tmp_path):
    with pytest.raises(ValueError, match="k_full_bit"):
        indexes.Chimera(
            index_folder=str(tmp_path),
            index_name="bad_kfb",
            k_refine=100,
            k_full_bit=500,
        )


def test_second_add_documents_raises(index, corpus):
    ids, documents = corpus
    with pytest.raises(ValueError, match="one shot"):
        index.add_documents(documents_ids=ids[:2], documents_embeddings=documents[:2])


def test_remove_documents_warns(index):
    with pytest.warns(UserWarning, match="does not support document removal"):
        index.remove_documents(["doc_0"])


def test_get_documents_embeddings_not_implemented(index):
    with pytest.raises(NotImplementedError):
        index.get_documents_embeddings([["doc_0"]])
