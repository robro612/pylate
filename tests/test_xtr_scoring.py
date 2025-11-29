"""Tests for XTR (eXact Token Retrieval) scoring."""

import pytest
import torch
from pylate.rank import score_xtr, RerankResult


class TestScoreXTR:
    """Test suite for the score_xtr function."""
    
    def test_basic_scoring_no_overlap(self):
        """Test basic case where each query token retrieves different documents."""
        query_doc_ids = [
            ["doc1", "doc2", "doc3"],  # Retrieved for query token 0
            ["doc4", "doc5", "doc6"],  # Retrieved for query token 1
        ]
        query_scores = [
            [0.9, 0.7, 0.5],  # Scores for query token 0
            [0.8, 0.6, 0.4],  # Scores for query token 1
        ]
        
        results = score_xtr(query_doc_ids, query_scores, k=6)
        
        # All 6 documents should be returned
        assert len(results) == 6
        
        # Each document should have exactly one high score and one minimum score
        # doc1: 0.9 (qt0) + 0.4 (min from qt1) = 1.3
        # doc2: 0.7 (qt0) + 0.4 (min from qt1) = 1.1
        # doc3: 0.5 (qt0) + 0.4 (min from qt1) = 0.9
        # doc4: 0.5 (min from qt0) + 0.8 (qt1) = 1.3
        # doc5: 0.5 (min from qt0) + 0.6 (qt1) = 1.1
        # doc6: 0.5 (min from qt0) + 0.4 (qt1) = 0.9
        
        # Results should be sorted by score
        assert results[0]["score"] == pytest.approx(1.3, abs=1e-5)
        assert results[0]["id"] in ["doc1", "doc4"]  # Both have score 1.3
        
    def test_overlapping_documents(self):
        """Test case where same document is retrieved for multiple query tokens."""
        query_doc_ids = [
            ["doc1", "doc2", "doc3"],  # Retrieved for query token 0
            ["doc2", "doc3", "doc4"],  # Retrieved for query token 1
        ]
        query_scores = [
            [0.9, 0.7, 0.5],  # Scores for query token 0
            [0.8, 0.6, 0.4],  # Scores for query token 1
        ]
        
        results = score_xtr(query_doc_ids, query_scores, k=4)
        
        # Should return 4 unique documents
        assert len(results) == 4
        
        # doc2 should have highest score: 0.7 (qt0) + 0.8 (qt1) = 1.5
        assert results[0]["id"] == "doc2"
        assert results[0]["score"] == pytest.approx(1.5, abs=1e-5)
        
        # doc1 should be second: 0.9 (qt0) + 0.4 (min from qt1) = 1.3
        assert results[1]["id"] == "doc1"
        assert results[1]["score"] == pytest.approx(1.3, abs=1e-5)
        
        # doc3 should be third: 0.5 (qt0) + 0.6 (qt1) = 1.1
        assert results[2]["id"] == "doc3"
        assert results[2]["score"] == pytest.approx(1.1, abs=1e-5)
        
        # doc4 should be fourth: 0.5 (min from qt0) + 0.4 (qt1) = 0.9
        assert results[3]["id"] == "doc4"
        assert results[3]["score"] == pytest.approx(0.9, abs=1e-5)
        
    def test_duplicate_doc_in_same_query_token(self):
        """Test that max is taken when same doc appears multiple times for same query token."""
        query_doc_ids = [
            ["doc1", "doc2", "doc1"],  # doc1 appears twice for qt0
            ["doc2", "doc3"],
        ]
        query_scores = [
            [0.9, 0.7, 0.6],  # doc1 has scores 0.9 and 0.6, should take max (0.9)
            [0.8, 0.5],
        ]
        
        results = score_xtr(query_doc_ids, query_scores, k=3)
        
        # doc1 should use max score 0.9 (not 0.6) for qt0
        # doc1: 0.9 (max from qt0) + 0.5 (min from qt1) = 1.4
        doc1_result = [r for r in results if r["id"] == "doc1"][0]
        assert doc1_result["score"] == pytest.approx(1.4, abs=1e-5)
        
    def test_integer_doc_ids(self):
        """Test that function works with integer document IDs."""
        query_doc_ids = [
            [1, 2, 3],
            [2, 3, 4],
        ]
        query_scores = [
            [0.9, 0.7, 0.5],
            [0.8, 0.6, 0.4],
        ]
        
        results = score_xtr(query_doc_ids, query_scores, k=4)
        
        # Should work the same as with string IDs
        assert len(results) == 4
        assert isinstance(results[0]["id"], int)
        
        # doc2 (id=2) should have highest score
        assert results[0]["id"] == 2
        assert results[0]["score"] == pytest.approx(1.5, abs=1e-5)
        
    def test_single_query_token(self):
        """Test with only one query token."""
        query_doc_ids = [
            ["doc1", "doc2", "doc3"],
        ]
        query_scores = [
            [0.9, 0.7, 0.5],
        ]
        
        results = score_xtr(query_doc_ids, query_scores, k=3)
        
        # With single query token, scores should equal original scores
        assert len(results) == 3
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == pytest.approx(0.9, abs=1e-5)
        assert results[1]["id"] == "doc2"
        assert results[1]["score"] == pytest.approx(0.7, abs=1e-5)
        
    def test_k_larger_than_unique_docs(self):
        """Test when k is larger than number of unique documents."""
        query_doc_ids = [
            ["doc1", "doc2"],
            ["doc2", "doc3"],
        ]
        query_scores = [
            [0.9, 0.7],
            [0.8, 0.5],
        ]
        
        results = score_xtr(query_doc_ids, query_scores, k=10)
        
        # Should only return 3 unique documents
        assert len(results) == 3
        
    def test_empty_query(self):
        """Test with empty query tokens."""
        query_doc_ids = []
        query_scores = []
        
        results = score_xtr(query_doc_ids, query_scores, k=5)
        
        assert len(results) == 0
        
    def test_cuda_device(self):
        """Test that cuda device parameter works (if CUDA available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        query_doc_ids = [
            ["doc1", "doc2"],
            ["doc2", "doc3"],
        ]
        query_scores = [
            [0.9, 0.7],
            [0.8, 0.5],
        ]
        
        results = score_xtr(query_doc_ids, query_scores, k=3, device="cuda")
        
        # Should get same results as CPU
        assert len(results) == 3
        assert results[0]["id"] == "doc2"
        
    def test_score_imputation(self):
        """Test that minimum scores are correctly used for imputation."""
        query_doc_ids = [
            ["doc1"],  # Only doc1 retrieved for qt0
            ["doc2"],  # Only doc2 retrieved for qt1
        ]
        query_scores = [
            [0.9],  # min = 0.9
            [0.5],  # min = 0.5
        ]
        
        results = score_xtr(query_doc_ids, query_scores, k=2)
        
        # doc1: 0.9 (qt0) + 0.5 (min from qt1) = 1.4
        # doc2: 0.9 (min from qt0) + 0.5 (qt1) = 1.4
        assert len(results) == 2
        assert results[0]["score"] == pytest.approx(1.4, abs=1e-5)
        assert results[1]["score"] == pytest.approx(1.4, abs=1e-5)
        
    def test_many_query_tokens(self):
        """Test with many query tokens."""
        n_tokens = 10
        query_doc_ids = [
            [f"doc{i}", f"doc{i+1}", f"doc{i+2}"]
            for i in range(n_tokens)
        ]
        query_scores = [
            [0.9, 0.7, 0.5]
            for _ in range(n_tokens)
        ]
        
        results = score_xtr(query_doc_ids, query_scores, k=5)
        
        # Should return top 5 documents
        assert len(results) == 5
        # All results should be valid
        for result in results:
            assert "id" in result
            assert "score" in result
            assert result["score"] > 0
            
    def test_result_structure(self):
        """Test that results have correct RerankResult structure."""
        query_doc_ids = [["doc1", "doc2"]]
        query_scores = [[0.9, 0.7]]
        
        results = score_xtr(query_doc_ids, query_scores, k=2)
        
        # Check structure
        assert isinstance(results, list)
        assert len(results) == 2
        
        for result in results:
            assert isinstance(result, dict)
            assert "id" in result
            assert "score" in result
            assert isinstance(result["score"], float)
            
    def test_descending_order(self):
        """Test that results are returned in descending score order."""
        query_doc_ids = [
            ["doc1", "doc2", "doc3", "doc4", "doc5"],
        ]
        query_scores = [
            [0.3, 0.9, 0.5, 0.1, 0.7],  # Deliberately unordered
        ]
        
        results = score_xtr(query_doc_ids, query_scores, k=5)
        
        # Check order
        assert results[0]["id"] == "doc2"  # 0.9
        assert results[1]["id"] == "doc5"  # 0.7
        assert results[2]["id"] == "doc3"  # 0.5
        assert results[3]["id"] == "doc1"  # 0.3
        assert results[4]["id"] == "doc4"  # 0.1
        
        # Verify scores are descending
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]


class TestScoreXTREdgeCases:
    """Test edge cases and corner cases."""
    
    def test_negative_scores(self):
        """Test handling of negative scores."""
        query_doc_ids = [
            ["doc1", "doc2"],
            ["doc2", "doc3"],
        ]
        query_scores = [
            [0.5, -0.3],  # Negative score
            [0.2, -0.5],
        ]
        
        results = score_xtr(query_doc_ids, query_scores, k=3)
        
        # Should handle negative scores correctly
        assert len(results) == 3
        # doc2: 0.5 (qt0, max of -0.3) + 0.2 (qt1) = 0.7
        # Actually: doc2 appears in both, so max(-0.3, min from qt0) and 0.2 from qt1
        
    def test_all_same_scores(self):
        """Test when all scores are identical."""
        query_doc_ids = [
            ["doc1", "doc2", "doc3"],
            ["doc4", "doc5", "doc6"],
        ]
        query_scores = [
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ]
        
        results = score_xtr(query_doc_ids, query_scores, k=6)
        
        # All documents should have same score
        assert len(results) == 6
        for result in results:
            assert result["score"] == pytest.approx(1.0, abs=1e-5)  # 0.5 + 0.5
            
    def test_very_small_k(self):
        """Test with k=1."""
        query_doc_ids = [
            ["doc1", "doc2", "doc3"],
            ["doc2", "doc3", "doc4"],
        ]
        query_scores = [
            [0.9, 0.7, 0.5],
            [0.8, 0.6, 0.4],
        ]
        
        results = score_xtr(query_doc_ids, query_scores, k=1)
        
        # Should return only top document
        assert len(results) == 1
        assert results[0]["id"] == "doc2"  # Highest combined score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

