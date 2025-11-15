"""Utility functions for token-level analysis and statistics."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm


class TokenTFIDFStats:
    """
    Collects corpus-level TF-IDF statistics at the token level.

    This class computes and stores:
    - Token frequencies across the corpus
    - Document frequencies for each token
    - IDF scores for each token
    - Per-document TF scores

    Useful for IDF-based token pruning strategies.

    Parameters
    ----------
    num_docs : int, optional
        Total number of documents in the corpus. If None, will be inferred from data.

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>>
    >>> # Tokenize documents
    >>> docs = ["This is document one", "This is document two"]
    >>> tokenized = [tokenizer.encode(doc) for doc in docs]
    >>>
    >>> # Collect stats
    >>> stats = TokenTFIDFStats()
    >>> stats.fit(tokenized)
    >>>
    >>> # Get IDF score for a token
    >>> token_id = tokenizer.convert_tokens_to_ids("document")
    >>> idf_score = stats.get_idf(token_id)
    >>>
    >>> # Get document-level TF for a token
    >>> tf_score = stats.get_tf(doc_idx=0, token_id=token_id)
    """

    def __init__(self, num_docs: Optional[int] = None):
        self.num_docs = num_docs
        self.fitted = False

        # Global statistics
        self.document_frequencies: dict[
            int, int
        ] = {}  # token_id -> number of docs containing it
        self.token_frequencies: dict[
            int, int
        ] = {}  # token_id -> total occurrences across corpus
        self.idf_scores: dict[int, float] = {}  # token_id -> IDF score

        # Per-document statistics
        self.doc_token_counts: list[
            dict[int, int]
        ] = []  # List of {token_id: count} per doc
        self.doc_lengths: list[int] = []  # Number of tokens per document

    @classmethod
    def from_colbert_model(
        cls,
        model: "ColBERT",
        documents: list[str],
        show_progress: bool = True,
    ) -> "TokenTFIDFStats":
        """
        Create TokenTFIDFStats from a ColBERT model and list of document strings.

        This is a convenience constructor that tokenizes documents using the model's
        tokenizer and then fits the statistics.

        Parameters
        ----------
        model : ColBERT
            ColBERT model to use for tokenization
        documents : list[str]
            List of document strings to process
        show_progress : bool, optional
            Whether to show progress during tokenization and fitting. Default is True.

        Returns
        -------
        TokenTFIDFStats
            Fitted TokenTFIDFStats instance

        Examples
        --------
        >>> from pylate.models import ColBERT
        >>> model = ColBERT("lightonai/GTE-ModernColBERT-v1")
        >>> docs = ["This is document one", "This is document two"]
        >>> stats = TokenTFIDFStats.from_colbert_model(model, docs)
        """
        # Tokenize documents
        tokenized_docs = []
        for doc_text in tqdm(
            documents,
            desc="Tokenizing documents",
            disable=not show_progress,
        ):
            tokens = model.tokenizer.encode(
                doc_text,
                add_special_tokens=True,
                truncation=True,
                max_length=model.document_length,
            )
            tokenized_docs.append(tokens)

        # Create instance and fit
        stats = cls(num_docs=len(documents))
        stats.fit(tokenized_docs, show_progress=show_progress)

        return stats

    def fit(
        self,
        tokenized_docs: List[Union[List[int], torch.Tensor, np.ndarray]],
        show_progress: bool = True,
    ) -> "TokenTFIDFStats":
        """
        Compute TF-IDF statistics from a list of tokenized documents.

        Parameters
        ----------
        tokenized_docs : List[Union[List[int], torch.Tensor, np.ndarray]]
            List of tokenized documents, where each document is represented as
            a list/array of token IDs.
        show_progress : bool, optional
            Whether to show a progress bar during processing. Default is True.

        Returns
        -------
        self : TokenTFIDFStats
            Returns self for method chaining.
        """
        if self.num_docs is None:
            self.num_docs = len(tokenized_docs)

        # Reset statistics
        self.document_frequencies = defaultdict(int)
        self.token_frequencies = defaultdict(int)
        self.doc_token_counts = []
        self.doc_lengths = []

        # First pass: collect token and document frequencies
        for doc_tokens in tqdm(
            tokenized_docs,
            desc="Computing TF-IDF statistics",
            disable=not show_progress,
        ):
            # Convert to list of ints if needed
            if isinstance(doc_tokens, torch.Tensor):
                doc_tokens = doc_tokens.cpu().tolist()
            elif isinstance(doc_tokens, np.ndarray):
                doc_tokens = doc_tokens.tolist()

            # Count tokens in this document
            token_counts = Counter(doc_tokens)
            self.doc_token_counts.append(dict(token_counts))
            self.doc_lengths.append(len(doc_tokens))

            # Update global statistics
            for token_id, count in token_counts.items():
                self.token_frequencies[token_id] += count
                self.document_frequencies[token_id] += 1

        # Compute IDF scores
        self._compute_idf()

        self.fitted = True
        return self

    def _compute_idf(self) -> None:
        """Compute IDF scores for all tokens."""
        for token_id, df in self.document_frequencies.items():
            # IDF = log(N / df) where N is total number of documents
            # Adding 1 to avoid division by zero and log(0)
            self.idf_scores[token_id] = math.log((self.num_docs + 1) / (df + 1))

    def get_idf(self, token_id: int) -> float:
        """
        Get the IDF score for a token.

        Parameters
        ----------
        token_id : int
            The token ID.

        Returns
        -------
        float
            IDF score. Returns a default high score for unseen tokens.
        """
        if not self.fitted:
            raise ValueError("Must call fit() before getting statistics")

        # Return a high IDF score for unseen tokens (they're rare)
        return self.idf_scores.get(token_id, math.log(self.num_docs + 1))

    def get_tf(self, doc_idx: int, token_id: int, normalized: bool = True) -> float:
        """
        Get the term frequency for a token in a specific document.

        Parameters
        ----------
        doc_idx : int
            The document index.
        token_id : int
            The token ID.
        normalized : bool, optional
            If True, returns TF normalized by document length (TF/doc_length).
            If False, returns raw count. Default is True.

        Returns
        -------
        float
            Term frequency score.
        """
        if not self.fitted:
            raise ValueError("Must call fit() before getting statistics")

        if doc_idx >= len(self.doc_token_counts):
            raise ValueError(f"Document index {doc_idx} out of range")

        count = self.doc_token_counts[doc_idx].get(token_id, 0)

        if normalized and self.doc_lengths[doc_idx] > 0:
            return count / self.doc_lengths[doc_idx]
        return float(count)

    def get_tfidf(self, doc_idx: int, token_id: int) -> float:
        """
        Get the TF-IDF score for a token in a specific document.

        Parameters
        ----------
        doc_idx : int
            The document index.
        token_id : int
            The token ID.

        Returns
        -------
        float
            TF-IDF score (TF * IDF).
        """
        tf = self.get_tf(doc_idx, token_id, normalized=True)
        idf = self.get_idf(token_id)
        return tf * idf

    def get_document_token_scores(
        self,
        doc_idx: int,
        score_type: str = "tfidf",
    ) -> dict[int, float]:
        """
        Get scores for all tokens in a document.

        Parameters
        ----------
        doc_idx : int
            The document index.
        score_type : str, optional
            Type of score to compute: 'tfidf', 'idf', or 'tf'. Default is 'tfidf'.

        Returns
        -------
        dict[int, float]
            Dictionary mapping token IDs to their scores.
        """
        if not self.fitted:
            raise ValueError("Must call fit() before getting statistics")

        if doc_idx >= len(self.doc_token_counts):
            raise ValueError(f"Document index {doc_idx} out of range")

        scores = {}
        for token_id in self.doc_token_counts[doc_idx].keys():
            if score_type == "tfidf":
                scores[token_id] = self.get_tfidf(doc_idx, token_id)
            elif score_type == "idf":
                scores[token_id] = self.get_idf(token_id)
            elif score_type == "tf":
                scores[token_id] = self.get_tf(doc_idx, token_id)
            else:
                raise ValueError(f"Unknown score_type: {score_type}")

        return scores

    def get_low_idf_tokens(
        self,
        threshold: Optional[float] = None,
        k: Optional[int] = None,
    ) -> list[int]:
        """
        Get tokens with low IDF scores (common tokens).

        Provide either threshold OR k, not both.

        Parameters
        ----------
        threshold : float, optional
            IDF threshold. Returns tokens with IDF < threshold.
        k : int, optional
            Number of tokens to return. Returns the k tokens with lowest IDF scores.

        Returns
        -------
        list[int]
            List of token IDs with low IDF scores.

        Examples
        --------
        >>> # Get tokens below threshold
        >>> common_tokens = stats.get_low_idf_tokens(threshold=0.5)
        >>>
        >>> # Get 100 most common tokens
        >>> top_100_common = stats.get_low_idf_tokens(k=100)
        """
        if not self.fitted:
            raise ValueError("Must call fit() before getting statistics")

        if threshold is None and k is None:
            raise ValueError("Must provide either 'threshold' or 'k'")

        if threshold is not None and k is not None:
            raise ValueError("Provide only one of 'threshold' or 'k', not both")

        if threshold is not None:
            return [
                token_id for token_id, idf in self.idf_scores.items() if idf < threshold
            ]

        # k is provided - return k tokens with lowest IDF
        sorted_tokens = sorted(self.idf_scores.items(), key=lambda x: x[1])
        return [token_id for token_id, _ in sorted_tokens[:k]]

    def get_high_idf_tokens(
        self,
        threshold: Optional[float] = None,
        k: Optional[int] = None,
    ) -> list[int]:
        """
        Get tokens with high IDF scores (rare tokens).

        Provide either threshold OR k, not both.

        Parameters
        ----------
        threshold : float, optional
            IDF threshold. Returns tokens with IDF > threshold.
        k : int, optional
            Number of tokens to return. Returns the k tokens with highest IDF scores.

        Returns
        -------
        list[int]
            List of token IDs with high IDF scores.

        Examples
        --------
        >>> # Get tokens above threshold
        >>> rare_tokens = stats.get_high_idf_tokens(threshold=2.0)
        >>>
        >>> # Get 100 rarest tokens
        >>> top_100_rare = stats.get_high_idf_tokens(k=100)
        """
        if not self.fitted:
            raise ValueError("Must call fit() before getting statistics")

        if threshold is None and k is None:
            raise ValueError("Must provide either 'threshold' or 'k'")

        if threshold is not None and k is not None:
            raise ValueError("Provide only one of 'threshold' or 'k', not both")

        if threshold is not None:
            return [
                token_id for token_id, idf in self.idf_scores.items() if idf > threshold
            ]

        # k is provided - return k tokens with highest IDF
        sorted_tokens = sorted(
            self.idf_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [token_id for token_id, _ in sorted_tokens[:k]]

    def get_corpus_statistics(self) -> dict:
        """
        Get summary statistics about the corpus.

        Returns
        -------
        dict
            Dictionary containing corpus-level statistics.
        """
        if not self.fitted:
            raise ValueError("Must call fit() before getting statistics")

        return {
            "num_documents": self.num_docs,
            "num_unique_tokens": len(self.token_frequencies),
            "total_tokens": sum(self.token_frequencies.values()),
            "avg_document_length": np.mean(self.doc_lengths),
            "median_document_length": np.median(self.doc_lengths),
            "min_document_length": min(self.doc_lengths),
            "max_document_length": max(self.doc_lengths),
            "avg_idf": np.mean(list(self.idf_scores.values())),
            "median_idf": np.median(list(self.idf_scores.values())),
        }

    def __repr__(self) -> str:
        if not self.fitted:
            return f"TokenTFIDFStats(fitted=False)"

        stats = self.get_corpus_statistics()
        return (
            f"TokenTFIDFStats(\n"
            f"  documents={stats['num_documents']},\n"
            f"  unique_tokens={stats['num_unique_tokens']},\n"
            f"  avg_doc_length={stats['avg_document_length']:.1f},\n"
            f"  avg_idf={stats['avg_idf']:.3f}\n"
            f")"
        )
