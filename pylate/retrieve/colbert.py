from __future__ import annotations

import logging
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from .. import indexes
from ..rank import RerankResult, rerank, score_xtr
from ..utils import iter_batch

logger = logging.getLogger(__name__)


class ColBERT:
    """ColBERT retriever.

    Parameters
    ----------
    index:
        The index to use for retrieval.

    Examples
    --------
    >>> from pylate import indexes, models, retrieve

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    ...     device="cpu",
    ... )

    >>> documents_ids = [f"document_id_{i}" for i in range(20)]
    >>> documents = [f"This is the content of document {i}." for i in range(20)]

    >>> documents_embeddings = model.encode(
    ...     sentences=documents,
    ...     batch_size=1,
    ...     is_query=False,
    ... )

    >>> index = indexes.PLAID(
    ...     index_folder="test_indexes",
    ...     index_name="colbert",
    ...     override=True,
    ... )

    >>> index = index.add_documents(
    ...     documents_ids=documents_ids,
    ...     documents_embeddings=documents_embeddings,
    ... )
    Computing centroids of embeddings.
    Creating FastPlaid index.

    >>> retriever = retrieve.ColBERT(index=index)

    >>> queries_embeddings = model.encode(
    ...     ["fruits are healthy.", "fruits are good for health."],
    ...     batch_size=1,
    ...     is_query=True,
    ... )

    >>> results = retriever.retrieve(
    ...     queries_embeddings=queries_embeddings,
    ...     k=2,
    ...     device="cpu",
    ... )

    >>> assert isinstance(results, list)
    >>> assert len(results) == 2

    >>> queries_embeddings = model.encode(
    ...     "fruits are healthy.",
    ...     batch_size=1,
    ...     is_query=True,
    ... )

    >>> results = retriever.retrieve(
    ...     queries_embeddings=queries_embeddings,
    ...     k=2,
    ...     device="cpu",
    ... )

    >>> assert isinstance(results, list)
    >>> assert len(results) == 1

    >>> results = retriever.retrieve(
    ...     queries_embeddings=queries_embeddings,
    ...     k=2,
    ...     device="cpu",
    ...     subset=["document_id_10"],
    ... )

    """

    def __init__(self, index: indexes.Base , verbose: bool = False) -> None:
        self.index = index
        self.verbose = verbose

    def retrieve(
        self,
        queries_embeddings: list[list | np.ndarray | torch.Tensor],
        k: int = 10,
        k_token: int = 100,
        device: str | None = None,
        batch_size: int = 1,
        subset: list[list[str]] | list[str] | None = None,
    ) -> list[list[RerankResult]]:
        """Retrieve documents for a list of queries.

        Parameters
        ----------
        queries_embeddings
            The queries embeddings.
        k
            The number of documents to retrieve.
        k_token
            The number of documents to retrieve from the index. Defaults to `k`.
        device
            The device to use for the embeddings. Defaults to queries_embeddings device.
        batch_size
            The batch size to use for retrieval.
        subset
            Optional subset of document IDs to restrict search to.
            Can be a single list (same filter for all queries) or
            list of lists (different filter per query).
            Document IDs should match the IDs used when adding documents.
            Only supported with PLAID index.

        """
        # PLAID index directly retrieves the documents
        if isinstance(self.index, indexes.PLAID):
            if self.verbose:
                logger.info("Retrieving documents with PLAID index")
            start_time = time.time()
            results = self.index(
                queries_embeddings=queries_embeddings,
                k=k,
                subset=subset,
            )
            if self.verbose:
                total_time = time.time() - start_time
                logger.info(
                    f"PLAID retrieval completed in {total_time:.4f}s ({total_time / len(queries_embeddings) * 1000:.2f}ms per query)"
                )
            return results

        # Other indexes first generate candidates by calling the index and then rerank them
        if k > k_token:
            logger.warning(
                f"k ({k}) is greater than k_token ({k_token}), setting k_token to k."
            )
            k_token = k

        total_retrieval_time = 0.0
        total_dedup_time = 0.0
        total_get_embeddings_time = 0.0
        total_rerank_time = 0.0
        num_batches = 0

        reranking_results = []
        for queries_embeddings_batch in tqdm(
            iter_batch(
                queries_embeddings,
                batch_size=batch_size,
            ),
            desc="Retrieving documents",
            total=len(queries_embeddings) // batch_size,
            disable=not self.verbose,
        ):
            # Initial retrieval from index
            retrieval_start = time.time()
            retrieved_elements = self.index(
                queries_embeddings=queries_embeddings_batch,
                k=k_token,
            )
            retrieval_time = time.time() - retrieval_start
            total_retrieval_time += retrieval_time

            # Deduplicate document IDs
            dedup_start = time.time()
            documents_ids = [
                list(
                    set(
                        [
                            document_id
                            for query_token_document_ids in query_documents_ids
                            for document_id in query_token_document_ids
                        ]
                    )
                )
                for query_documents_ids in retrieved_elements["documents_ids"]
            ]
            dedup_time = time.time() - dedup_start
            total_dedup_time += dedup_time

            # Get document embeddings for reranking
            get_embeddings_start = time.time()
            documents_embeddings = self.index.get_documents_embeddings(documents_ids)
            get_embeddings_time = time.time() - get_embeddings_start
            total_get_embeddings_time += get_embeddings_time

            # Rerank
            rerank_start = time.time()
            batch_rerank_results = rerank(
                documents_ids=documents_ids,
                queries_embeddings=queries_embeddings_batch,
                documents_embeddings=documents_embeddings,
                device=device,
            )
            rerank_time = time.time() - rerank_start
            total_rerank_time += rerank_time

            reranking_results.extend(batch_rerank_results)
            num_batches += 1

        # Log timing breakdown if verbose
        if self.verbose:
            total_time = (
                total_retrieval_time
                + total_dedup_time
                + total_get_embeddings_time
                + total_rerank_time
            )
            logger.info(
                f"Retrieval timing breakdown (total: {total_time:.4f}s, {num_batches} batches):"
            )
            logger.info(
                f"  - Initial retrieval: {total_retrieval_time:.4f}s ({total_retrieval_time / total_time * 100:.1f}%)"
            )
            logger.info(
                f"  - Deduplication:     {total_dedup_time:.4f}s ({total_dedup_time / total_time * 100:.1f}%)"
            )
            logger.info(
                f"  - Get embeddings:    {total_get_embeddings_time:.4f}s ({total_get_embeddings_time / total_time * 100:.1f}%)"
            )
            logger.info(
                f"  - Reranking:         {total_rerank_time:.4f}s ({total_rerank_time / total_time * 100:.1f}%)"
            )

        return [query_results[:k] for query_results in reranking_results]

    def retrieve_xtr(
        self,
        queries_embeddings: list[list | np.ndarray | torch.Tensor],
        k: int = 10,
        k_token: int = 40_000,
        device: str | None = None,
        batch_size: int = 1,
        subset: list[list[str]] | list[str] | None = None,
    ) -> list[list[RerankResult]]:
        """Retrieve documents using XTR (eXact Token Retrieval) scoring.
        
        XTR differs from standard ColBERT retrieval in that it doesn't do a full
        reranking step. Instead, it only scores documents using initially retrieved
        tokens and imputes missing scores with the minimum score per query token.
        
        Parameters
        ----------
        queries_embeddings
            The queries embeddings.
        k
            The number of documents to retrieve.
        k_token
            The number of documents to retrieve from the index per query token.
        device
            The device to use for computation. Defaults to 'cpu'.
        batch_size
            The batch size to use for retrieval.
        subset
            Optional subset of document IDs to restrict search to.
            Only supported with certain index types.
        
        Returns
        -------
        list[list[RerankResult]]
            List of results for each query, where each result contains
            document IDs and scores sorted by score (descending).
        
        """
        if device is None:
            device = 'cpu'
        
        total_retrieval_time = 0.0
        total_scoring_time = 0.0
        num_batches = 0
        
        results = []
        
        progress_bar = tqdm(
            iter_batch(queries_embeddings, batch_size=batch_size),
            desc="Retrieving documents (XTR)",
            disable=not self.verbose,
        )
        for batch_queries_embeddings in progress_bar:
            # Initial retrieval from index
            retrieval_start = time.time()
            index_results = self.index(batch_queries_embeddings, k=k_token, subset=subset)
            retrieval_time = time.time() - retrieval_start
            total_retrieval_time += retrieval_time
            
            # XTR scoring
            scoring_start = time.time()
            for query_doc_ids, query_scores in zip(
                index_results["documents_ids"], 
                index_results["distances"]
            ):
                # Use the score_xtr helper function to compute XTR scores
                query_results = score_xtr(
                    query_doc_ids=query_doc_ids,
                    query_scores=query_scores,
                    k=k,
                    device=device,
                )
                
                results.append(query_results)
            scoring_time = time.time() - scoring_start
            total_scoring_time += scoring_time
            num_batches += 1

            batch_count = max(1, len(batch_queries_embeddings))
            per_query_retrieval = retrieval_time / batch_count  # seconds per query
            per_query_scoring = scoring_time / batch_count      # seconds per query
            per_query_total = per_query_retrieval + per_query_scoring  # seconds per query
            if not progress_bar.disable:
                progress_bar.set_postfix(
                    {
                        "per_query_retrieval (s)": f"{per_query_retrieval:.3f} ({per_query_retrieval / (per_query_total + 1e-12) * 100:.1f}%)",
                        "per_query_scoring (s)": f"{per_query_scoring:.3f} ({per_query_scoring / (per_query_total + 1e-12) * 100:.1f}%)",
                        "per_query_total (s)": f"{per_query_total:.3f}",
                    }
                )
        
        # Log timing breakdown if verbose
        if self.verbose:
            total_time = total_retrieval_time + total_scoring_time
            logger.info(
                f"XTR retrieval timing breakdown (total: {total_time:.4f}s, {num_batches} batches of {batch_size} queries):"
            )
            logger.info(
                f"  - Index retrieval: {total_retrieval_time:.4f}s ({total_retrieval_time / total_time * 100:.1f}%)"
            )
            logger.info(
                f"  - XTR scoring:      {total_scoring_time:.4f}s ({total_scoring_time / total_time * 100:.1f}%)"
            )
            if len(queries_embeddings) > 0:
                logger.info(
                    f"  - Per query:        {total_time / len(queries_embeddings) * 1000:.2f}ms"
                )
        
        return results


                