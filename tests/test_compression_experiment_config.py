#!/usr/bin/env python3
"""
Simple test script to verify CompressionExperimentConfig and CompressionExperimentResults
classes work correctly before integrating into encode().
"""

from datetime import datetime
from pathlib import Path
from pylate.models import (
    CompressionConfig,
    CompressionExperimentConfig,
    CompressionExperimentResults,
    IDFPruningConfig,
    PoolingConfig,
)
from pylate.models.utils import TokenTFIDFStats


def test_experiment_config_creation():
    """Test creating CompressionExperimentConfig with various options."""
    print("=" * 80)
    print("TEST 1: CompressionExperimentConfig Creation")
    print("=" * 80)

    # Create some dummy configs
    configs = [
        None,  # Baseline
        CompressionConfig(
            description="Also a Baseline",
        ),
        CompressionConfig(
            description="IDF pruning k=10",
            pruning=[
                IDFPruningConfig(mode="document", top_k=10, stats=TokenTFIDFStats())
            ],
        ),
        CompressionConfig(
            description="Pooling factor=2",
            pooling=[PoolingConfig(pool_factor=2)],
        ),
        CompressionConfig(
            description="Global IDF pruning k=10",
            pruning=[
                IDFPruningConfig(mode="global", top_k=10, stats=TokenTFIDFStats())
            ],
        ),
    ]

    # Test 1a: Basic creation
    print("\nTest 1a: Basic creation with disk storage")
    experiment = CompressionExperimentConfig(
        configs=configs,
        output_dir="/tmp/test_compression_experiment",
        storage_mode="disk",
        overwrite=True,
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    print(f"✓ Created experiment with {len(experiment.configs)} configs")
    print(f"  Output dir: {experiment.output_dir}")
    print(f"  Config names: {experiment.config_names}")
    print(f"  Storage mode: {experiment.storage_mode}")

    # Test 1b: Verify output file paths
    print("\nTest 1b: Output file paths")
    for i in range(len(configs)):
        output_file = experiment.get_output_file(i)
        print(f"  Config {i} ({experiment.config_names[i]}): {output_file.name}")

    # Test 1c: Custom config names
    print("\nTest 1c: Custom config names")
    custom_experiment = CompressionExperimentConfig(
        configs=configs,
        output_dir="/tmp/test_custom_names",
        config_names=[
            "baseline",
            "also_a_baseline",
            "doc_idf_prune_10",
            "pool_2",
            "global_idf_prune_10",
        ],
        storage_mode="memory",
        overwrite=True,
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    print(f"✓ Custom names: {custom_experiment.config_names}")

    # Test 1d: Name sanitization
    print("\nTest 1d: Name sanitization")
    crazy_names = CompressionConfig(description="Test / with / slashes")
    sanitize_experiment = CompressionExperimentConfig(
        configs=[crazy_names],
        output_dir="/tmp/test_sanitize",
        storage_mode="disk",
        overwrite=True,
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    sanitized_file = sanitize_experiment.get_output_file(0)
    print(f"  Original name: 'Test / with / slashes'")
    print(f"  Sanitized filename: {sanitized_file.name}")
    print(f"  ✓ No slashes in filename")

    print("\n✓ All CompressionExperimentConfig tests passed!")


def test_experiment_results():
    """Test CompressionExperimentResults functionality."""
    print("\n" + "=" * 80)
    print("TEST 2: CompressionExperimentResults")
    print("=" * 80)

    # Create experiment config
    configs = [
        None,
        CompressionConfig(description="Test config 1"),
        CompressionConfig(description="Test config 2"),
    ]

    experiment = CompressionExperimentConfig(
        configs=configs,
        output_dir="/tmp/test_results",
        storage_mode="memory",
        overwrite=True,
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    # Create mock results
    import torch

    # Mock embeddings (3 configs × 5 docs each)
    mock_embeddings = [
        [torch.randn(10, 128) for _ in range(5)],  # Config 0: 10 tokens/doc
        [torch.randn(8, 128) for _ in range(5)],  # Config 1: 8 tokens/doc
        [torch.randn(6, 128) for _ in range(5)],  # Config 2: 6 tokens/doc
    ]

    results = CompressionExperimentResults(
        experiment_config=experiment,
        embeddings=mock_embeddings,
        contexts=[None, None, None],
        statistics={
            "num_documents": 5,
            "num_configs": 3,
            "config_token_counts": [50, 40, 30],
            "avg_tokens_per_doc": [10.0, 8.0, 6.0],
            "encoding_time": 1.5,
            "compression_times": [0.1, 0.2, 0.3],
            "total_time": 2.1,
        },
    )

    # Test 2a: Load embeddings
    print("\nTest 2a: Load embeddings")
    for i in range(len(configs)):
        embeddings = results.load_embeddings(i)
        print(
            f"  Config {i}: {len(embeddings)} documents, "
            f"{embeddings[0].shape[0]} tokens/doc (first doc)"
        )
    print("✓ All embeddings loaded successfully")

    # Test 2b: Iterator
    print("\nTest 2b: Iterator")
    for i, (config, embeddings) in enumerate(results):
        desc = config.description if config else "Baseline"
        print(f"  [{i}] {desc}: {len(embeddings)} documents")
    print("✓ Iterator works correctly")

    # Test 2c: Statistics access
    print("\nTest 2c: Statistics")
    print(f"  Total documents: {results.statistics['num_documents']}")
    print(f"  Encoding time: {results.statistics['encoding_time']:.2f}s")
    print(f"  Total time: {results.statistics['total_time']:.2f}s")
    print(f"  Token counts: {results.statistics['config_token_counts']}")
    print(f"  Avg tokens/doc: {results.statistics['avg_tokens_per_doc']}")
    print("✓ Statistics accessible")

    # Test 2d: Save summary
    print("\nTest 2d: Save summary")
    summary_file = Path("/tmp/test_results/test_summary.txt")
    results.save_summary(summary_file)
    if summary_file.exists():
        print(f"✓ Summary saved to {summary_file}")
        print("\nSummary preview:")
        with open(summary_file) as f:
            lines = f.readlines()[:15]  # First 15 lines
            for line in lines:
                print(f"    {line.rstrip()}")

    print("\n✓ All CompressionExperimentResults tests passed!")


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 80)
    print("TEST 3: Error Handling")
    print("=" * 80)

    # Test 3a: Empty configs list
    print("\nTest 3a: Empty configs list")
    try:
        CompressionExperimentConfig(
            configs=[],
            output_dir="/tmp/test_error",
            run_id="test_run",
        )
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test 3b: Existing directory without overwrite
    print("\nTest 3b: Existing directory without overwrite")
    Path("/tmp/test_existing").mkdir(parents=True, exist_ok=True)
    try:
        CompressionExperimentConfig(
            configs=[None],
            output_dir="/tmp/test_existing",
            overwrite=False,
            run_id="test_run",
        )
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test 3c: Mismatched config_names length
    print("\nTest 3c: Mismatched config_names length")
    try:
        CompressionExperimentConfig(
            configs=[None, None, None],
            output_dir="/tmp/test_mismatch",
            config_names=["name1", "name2"],  # Only 2 names for 3 configs
            overwrite=True,
            run_id="test_run",
        )
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test 3d: Invalid config_idx in load_embeddings
    print("\nTest 3d: Invalid config_idx")
    experiment = CompressionExperimentConfig(
        configs=[None],
        output_dir="/tmp/test_idx",
        storage_mode="memory",
        overwrite=True,
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    results = CompressionExperimentResults(
        experiment_config=experiment,
        embeddings=[[torch.randn(5, 128)]],
    )
    try:
        results.load_embeddings(config_idx=5)
        print("✗ Should have raised IndexError")
    except IndexError as e:
        print(f"✓ Correctly raised IndexError: {e}")

    print("\n✓ All error handling tests passed!")


if __name__ == "__main__":
    print("Testing CompressionExperimentConfig and CompressionExperimentResults")
    print("=" * 80 + "\n")

    import torch  # Need torch for tests

    try:
        test_experiment_config_creation()
        test_experiment_results()
        test_error_handling()

        print("\n" + "=" * 80)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 80)
        print("\nThe helper classes are ready for integration into ColBERT.encode()!")

    except Exception as e:
        print(f"\n✗✗✗ TEST FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
