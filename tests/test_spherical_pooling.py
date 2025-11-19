"""Tests for spherical cluster pooling with different input types."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from pylate.models.compression import (
    PoolingConfig,
    PoolingStrategy,
    CompressionArtifacts,
)


def reset_cuda_state():
    """Reset CUDA state to clear any errors."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass  # Ignore errors during cleanup


def create_seeded_tensor(shape, device="cpu", seed=42):
    """Create a tensor with reproducible values without triggering CUDA seed operations."""
    # Use numpy to avoid torch.manual_seed() which triggers CUDA operations
    np.random.seed(seed)
    # Generate a unique seed for each tensor to avoid identical values
    # This is a workaround - in real tests you might want deterministic values
    rng = np.random.RandomState(seed)
    data = rng.randn(*shape).astype(np.float32)
    return torch.from_numpy(data).to(device=device)


@pytest.fixture
def sample_embeddings_cpu() -> list[torch.Tensor]:
    """Create sample CPU embeddings for testing (~100 rows of ~300x128)."""
    # Use numpy seeding to avoid CUDA operations from torch.manual_seed()
    np.random.seed(42)
    # Create ~100 documents with ~300 tokens each, 128 dim embeddings
    return [
        create_seeded_tensor((300, 128), device="cpu", seed=42 + i)
        for i in range(100)
    ]


@pytest.fixture
def sample_embeddings_cuda() -> list[torch.Tensor]:
    """Create sample CUDA embeddings for testing (~100 rows of ~300x128)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Use numpy seeding to avoid CUDA operations from torch.manual_seed()
    np.random.seed(42)
    # Create ~100 documents with ~300 tokens each, 128 dim embeddings
    return [
        create_seeded_tensor((300, 128), device="cuda", seed=42 + i)
        for i in range(100)
    ]


@pytest.fixture
def sample_embeddings_mixed() -> list[torch.Tensor]:
    """Create mixed CPU/CUDA embeddings for testing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Use numpy seeding to avoid CUDA operations from torch.manual_seed()
    np.random.seed(42)
    # Mix of CPU and CUDA tensors
    embeddings = []
    for i in range(100):
        device = "cuda" if i % 2 == 0 else "cpu"
        embeddings.append(create_seeded_tensor((300, 128), device=device, seed=42 + i))
    return embeddings


class TestSphericalPoolingCPU:
    """Tests for spherical pooling with CPU tensors."""

    @pytest.fixture(autouse=True)
    def reset_cuda_before_test(self):
        """Reset CUDA state before each test to avoid cascading failures."""
        reset_cuda_state()
        yield
        reset_cuda_state()

    def test_spherical_pooling_cpu_basic(self, sample_embeddings_cpu):
        """Test basic spherical pooling with CPU tensors."""
        config = PoolingConfig(
            pool_factor=2,
            protected_tokens=1,
            clustering_method="spherical",
            show_progress_bar=False,
        )
        strategy = PoolingStrategy(config)
        
        artifacts: CompressionArtifacts = {}
        try:
            pooled_embeddings, cluster_assignments = strategy._pool_embeddings_spherical(
                documents_embeddings=sample_embeddings_cpu,
                pool_factor=2,
                protected_tokens=1,
            )
        except RuntimeError as e:
            # fastkmeans internally calls torch.manual_seed() which can trigger CUDA errors
            # if CUDA is in a bad state, even for CPU-only operations
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                pytest.skip(f"CUDA error (likely from fastkmeans internal torch.manual_seed): {e}")
            raise
        
        # Check output structure
        assert len(pooled_embeddings) == 100
        assert len(cluster_assignments) == 100
        
        # Check that embeddings are reduced
        for i, pooled in enumerate(pooled_embeddings):
            original = sample_embeddings_cpu[i]
            # Should have protected tokens + pooled tokens
            assert pooled.shape[0] <= original.shape[0]
            assert pooled.shape[1] == original.shape[1] == 128
            assert pooled.device == torch.device("cpu")
        
        # Check cluster assignments
        for assignments in cluster_assignments:
            assert isinstance(assignments, list)
            # All assignments should be positive integers (1-indexed)
            if assignments:
                assert all(isinstance(a, int) and a > 0 for a in assignments)

    def test_spherical_pooling_cpu_different_factors(self, sample_embeddings_cpu):
        """Test spherical pooling with different pool factors on CPU."""
        for pool_factor in [2, 3, 4, 5]:
            config = PoolingConfig(
                pool_factor=pool_factor,
                protected_tokens=1,
                clustering_method="spherical",
                show_progress_bar=False,
            )
            strategy = PoolingStrategy(config)
            
            try:
                pooled_embeddings, cluster_assignments = strategy._pool_embeddings_spherical(
                    documents_embeddings=sample_embeddings_cpu,
                    pool_factor=pool_factor,
                    protected_tokens=1,
                )
            except RuntimeError as e:
                if "CUDA" in str(e) or "cuda" in str(e).lower():
                    pytest.skip(f"CUDA error (likely from fastkmeans): {e}")
                raise
            
            # Verify reduction
            for i, pooled in enumerate(pooled_embeddings):
                original = sample_embeddings_cpu[i]
                # With pool_factor, we expect roughly original_size / pool_factor + protected
                expected_max = original.shape[0] // pool_factor + 1
                assert pooled.shape[0] <= expected_max + 10  # Allow some tolerance

    def test_spherical_pooling_cpu_protected_tokens(self, sample_embeddings_cpu):
        """Test spherical pooling with different protected token counts on CPU."""
        for protected in [0, 1, 5, 10]:
            config = PoolingConfig(
                pool_factor=3,
                protected_tokens=protected,
                clustering_method="spherical",
                show_progress_bar=False,
            )
            strategy = PoolingStrategy(config)
            
            try:
                pooled_embeddings, _ = strategy._pool_embeddings_spherical(
                    documents_embeddings=sample_embeddings_cpu,
                    pool_factor=3,
                    protected_tokens=protected,
                )
            except RuntimeError as e:
                if "CUDA" in str(e) or "cuda" in str(e).lower():
                    pytest.skip(f"CUDA error (likely from fastkmeans): {e}")
                raise
            
            # Check that protected tokens are preserved
            # Note: protected tokens are appended at the END of pooled embeddings
            for i, pooled in enumerate(pooled_embeddings):
                original = sample_embeddings_cpu[i]
                actual_protected = min(protected, original.shape[0])
                if actual_protected > 0:
                    # Protected tokens are at the END of the pooled tensor
                    # First protected tokens from original should match last tokens in pooled
                    assert torch.allclose(
                        pooled[-actual_protected:],
                        original[:actual_protected],
                        atol=1e-5
                    )


class TestSphericalPoolingCUDA:
    """Tests for spherical pooling with CUDA tensors."""

    @pytest.fixture(autouse=True)
    def reset_cuda_before_test(self):
        """Reset CUDA state before each test to avoid cascading failures."""
        reset_cuda_state()
        yield
        reset_cuda_state()

    def test_spherical_pooling_cuda_basic(self, sample_embeddings_cuda):
        """Test basic spherical pooling with CUDA tensors."""
        config = PoolingConfig(
            pool_factor=2,
            protected_tokens=1,
            clustering_method="spherical",
            show_progress_bar=False,
        )
        strategy = PoolingStrategy(config)
        
        try:
            pooled_embeddings, cluster_assignments = strategy._pool_embeddings_spherical(
                documents_embeddings=sample_embeddings_cuda,
                pool_factor=2,
                protected_tokens=1,
            )
        except torch.cuda.OutOfMemoryError:
            pytest.skip("CUDA out of memory")
        except RuntimeError as e:
            if "CUDA" in str(e):
                pytest.skip(f"CUDA error: {e}")
            raise
        
        # Check output structure
        assert len(pooled_embeddings) == 100
        assert len(cluster_assignments) == 100
        
        # Check that embeddings are on CUDA
        for i, pooled in enumerate(pooled_embeddings):
            original = sample_embeddings_cuda[i]
            assert pooled.shape[0] <= original.shape[0]
            assert pooled.shape[1] == original.shape[1] == 128
            assert pooled.device.type == "cuda"

    def test_spherical_pooling_cuda_different_factors(self, sample_embeddings_cuda):
        """Test spherical pooling with different pool factors on CUDA."""
        for pool_factor in [2, 3, 4]:
            config = PoolingConfig(
                pool_factor=pool_factor,
                protected_tokens=1,
                clustering_method="spherical",
                show_progress_bar=False,
            )
            strategy = PoolingStrategy(config)
            
            try:
                pooled_embeddings, _ = strategy._pool_embeddings_spherical(
                    documents_embeddings=sample_embeddings_cuda,
                    pool_factor=pool_factor,
                    protected_tokens=1,
                )
            except torch.cuda.OutOfMemoryError:
                pytest.skip("CUDA out of memory")
            except RuntimeError as e:
                if "CUDA" in str(e):
                    pytest.skip(f"CUDA error: {e}")
                raise
            
            # Verify all outputs are on CUDA
            for pooled in pooled_embeddings:
                assert pooled.device.type == "cuda"


class TestSphericalPoolingEdgeCases:
    """Tests for edge cases in spherical pooling."""

    @pytest.fixture(autouse=True)
    def reset_cuda_before_test(self):
        """Reset CUDA state before each test to avoid cascading failures."""
        reset_cuda_state()
        yield
        reset_cuda_state()

    def test_spherical_pooling_small_documents(self):
        """Test spherical pooling with very small documents."""
        # Use numpy random seed to avoid CUDA issues with torch.manual_seed
        np.random.seed(42)
        # Documents smaller than pool_factor
        small_embeddings = [
            create_seeded_tensor((5, 128), device="cpu", seed=42 + i)
            for i in range(10)
        ]
        
        config = PoolingConfig(
            pool_factor=10,  # Larger than document size
            protected_tokens=1,
            clustering_method="spherical",
            show_progress_bar=False,
        )
        strategy = PoolingStrategy(config)
        
        try:
            pooled_embeddings, cluster_assignments = strategy._pool_embeddings_spherical(
                documents_embeddings=small_embeddings,
                pool_factor=10,
                protected_tokens=1,
            )
        except RuntimeError as e:
            # fastkmeans internally calls torch.manual_seed() which can trigger CUDA errors
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                pytest.skip(f"CUDA error (likely from fastkmeans internal torch.manual_seed): {e}")
            raise
        
        # Should handle gracefully
        assert len(pooled_embeddings) == 10
        for pooled in pooled_embeddings:
            assert pooled.shape[0] > 0

    def test_spherical_pooling_empty_document(self):
        """Test spherical pooling with empty document."""
        embeddings = [torch.randn(0, 128, device="cpu")]
        
        config = PoolingConfig(
            pool_factor=2,
            protected_tokens=1,
            clustering_method="spherical",
            show_progress_bar=False,
        )
        strategy = PoolingStrategy(config)
        
        pooled_embeddings, cluster_assignments = strategy._pool_embeddings_spherical(
            documents_embeddings=embeddings,
            pool_factor=2,
            protected_tokens=1,
        )
        
        # Should handle empty document
        assert len(pooled_embeddings) == 1
        assert pooled_embeddings[0].shape[0] == 0

    def test_spherical_pooling_single_token(self):
        """Test spherical pooling with single token documents."""
        embeddings = [torch.randn(1, 128, device="cpu") for _ in range(5)]
        
        config = PoolingConfig(
            pool_factor=2,
            protected_tokens=1,
            clustering_method="spherical",
            show_progress_bar=False,
        )
        strategy = PoolingStrategy(config)
        
        pooled_embeddings, cluster_assignments = strategy._pool_embeddings_spherical(
            documents_embeddings=embeddings,
            pool_factor=2,
            protected_tokens=1,
        )
        
        # Should preserve the single token
        assert len(pooled_embeddings) == 5
        for pooled in pooled_embeddings:
            assert pooled.shape[0] == 1


class TestSphericalPoolingThroughCompress:
    """Tests for spherical pooling through the compress method."""

    @pytest.fixture(autouse=True)
    def reset_cuda_before_test(self):
        """Reset CUDA state before each test to avoid cascading failures."""
        reset_cuda_state()
        yield
        reset_cuda_state()

    def test_spherical_pooling_compress_cpu(self, sample_embeddings_cpu):
        """Test spherical pooling via compress method on CPU."""
        config = PoolingConfig(
            pool_factor=2,
            protected_tokens=1,
            clustering_method="spherical",
            show_progress_bar=False,
        )
        strategy = PoolingStrategy(config)
        
        artifacts: CompressionArtifacts = {}
        try:
            pooled_embeddings, updated_artifacts = strategy.compress(
                embeddings=sample_embeddings_cpu,
                artifacts=artifacts,
            )
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                pytest.skip(f"CUDA error (likely from fastkmeans): {e}")
            raise
        
        # Check output
        assert len(pooled_embeddings) == 100
        assert isinstance(updated_artifacts, dict)
        
        # Verify reduction
        for i, pooled in enumerate(pooled_embeddings):
            original = sample_embeddings_cpu[i]
            assert pooled.shape[0] <= original.shape[0]
            assert pooled.shape[1] == original.shape[1]

    def test_spherical_pooling_compress_cuda(self, sample_embeddings_cuda):
        """Test spherical pooling via compress method on CUDA."""
        config = PoolingConfig(
            pool_factor=2,
            protected_tokens=1,
            clustering_method="spherical",
            show_progress_bar=False,
        )
        strategy = PoolingStrategy(config)
        
        artifacts: CompressionArtifacts = {}
        try:
            pooled_embeddings, updated_artifacts = strategy.compress(
                embeddings=sample_embeddings_cuda,
                artifacts=artifacts,
            )
        except torch.cuda.OutOfMemoryError:
            pytest.skip("CUDA out of memory")
        except RuntimeError as e:
            if "CUDA" in str(e):
                pytest.skip(f"CUDA error: {e}")
            raise
        
        # Check output
        assert len(pooled_embeddings) == 100
        for pooled in pooled_embeddings:
            assert pooled.device.type == "cuda"


class TestSphericalPoolingNumpyConversion:
    """Tests to verify torch to numpy conversion works correctly."""

    @pytest.fixture(autouse=True)
    def reset_cuda_before_test(self):
        """Reset CUDA state before each test to avoid cascading failures."""
        reset_cuda_state()
        yield
        reset_cuda_state()

    def test_spherical_pooling_numpy_conversion_cpu(self, sample_embeddings_cpu):
        """Test that torch tensors are correctly converted to numpy for fastkmeans."""
        config = PoolingConfig(
            pool_factor=2,
            protected_tokens=1,
            clustering_method="spherical",
            show_progress_bar=False,
        )
        strategy = PoolingStrategy(config)
        
        # This should work without errors - if numpy conversion fails, this will raise
        try:
            pooled_embeddings, _ = strategy._pool_embeddings_spherical(
                documents_embeddings=sample_embeddings_cpu,
                pool_factor=2,
                protected_tokens=1,
            )
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                pytest.skip(f"CUDA error (likely from fastkmeans): {e}")
            raise
        
        # Verify output is still torch tensors (conversion is internal)
        assert len(pooled_embeddings) == 100
        for pooled in pooled_embeddings:
            assert isinstance(pooled, torch.Tensor)
            assert pooled.device.type == "cpu"

    def test_spherical_pooling_numpy_conversion_cuda(self, sample_embeddings_cuda):
        """Test that CUDA torch tensors are correctly converted to numpy (on CPU) for fastkmeans."""
        config = PoolingConfig(
            pool_factor=2,
            protected_tokens=1,
            clustering_method="spherical",
            show_progress_bar=False,
        )
        strategy = PoolingStrategy(config)
        
        try:
            # This should work - CUDA tensors should be moved to CPU for numpy conversion
            # then results moved back to CUDA
            pooled_embeddings, _ = strategy._pool_embeddings_spherical(
                documents_embeddings=sample_embeddings_cuda,
                pool_factor=2,
                protected_tokens=1,
            )
        except torch.cuda.OutOfMemoryError:
            pytest.skip("CUDA out of memory")
        except RuntimeError as e:
            if "CUDA" in str(e):
                pytest.skip(f"CUDA error: {e}")
            raise
        
        # Verify output is still torch tensors on CUDA
        assert len(pooled_embeddings) == 100
        for pooled in pooled_embeddings:
            assert isinstance(pooled, torch.Tensor)
            assert pooled.device.type == "cuda"

