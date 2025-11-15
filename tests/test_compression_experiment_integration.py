#!/usr/bin/env python3
"""
Test the integrated CompressionExperimentConfig functionality in ColBERT.encode().
"""

from datetime import datetime
from pathlib import Path
import torch
from pylate.models import (
    ColBERT,
    CompressionConfig,
    CompressionExperimentConfig,
    CompressionExperimentResults,
    IDFPruningConfig,
    PoolingConfig,
)
from pylate.models.utils import TokenTFIDFStats
from pylate import evaluation


def setup_model_and_data():
    """
    Initialize model and load test dataset.

    Returns
    -------
    tuple
        (model, documents, idf_stats)
    """
    print("=" * 80)
    print("Setup: Loading model and data")
    print("=" * 80)

    # Create model
    print("\nInitializing model...")
    model = ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        document_length=300,
        query_length=32,
        device="cuda",
    )

    # Load dataset
    print("Loading BEIR nfcorpus dataset...")
    documents, queries, qrels = evaluation.load_beir(
        dataset_name="nfcorpus",
        split="test",
    )
    documents = [doc["text"] for doc in documents[:2000]]
    print(f"✓ Loaded {len(documents)} documents")

    # Collect IDF stats
    print("\nCollecting IDF statistics...")
    tokenized_docs = []
    for doc in documents:
        tokens = model.tokenizer.encode(
            doc,
            add_special_tokens=True,
            truncation=True,
            max_length=model.document_length,
        )
        tokenized_docs.append(tokens)

    idf_stats = TokenTFIDFStats()
    idf_stats.fit(tokenized_docs, show_progress=False)
    print(f"✓ Collected stats for {len(idf_stats.idf_scores)} unique tokens")

    return model, documents, idf_stats


def run_compression_experiment(
    model: ColBERT,
    documents: list[str],
    experiment_config: CompressionExperimentConfig,
    batch_size: int = 1024,
    show_progress: bool = True,
) -> CompressionExperimentResults:
    """
    Run a compression experiment with multiple configs.

    Parameters
    ----------
    model : ColBERT
        The model to use for encoding
    documents : list[str]
        Documents to encode
    experiment_config : CompressionExperimentConfig
        Experiment configuration with multiple compression configs
    batch_size : int
        Batch size for encoding
    show_progress : bool
        Whether to show progress bar

    Returns
    -------
    CompressionExperimentResults
        Results containing embeddings and statistics for all configs
    """
    print(f"\nRunning experiment with {len(experiment_config.configs)} configs...")
    print(f"  Storage mode: {experiment_config.storage_mode}")
    print(f"  Configs: {experiment_config.config_names}")

    results = model.encode(
        documents,
        is_query=False,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        compression_config=experiment_config,
    )

    print(f"✓ Experiment complete!")
    return results


def test_results_structure(
    results: CompressionExperimentResults, storage_mode: str, model_device: str = "cuda"
):
    """Test that results have the expected structure and device placement."""
    print("\n" + "=" * 80)
    print(f"TEST: Results Structure (storage_mode={storage_mode})")
    print("=" * 80)

    # Check required attributes
    assert hasattr(results, "experiment_config"), "Missing experiment_config"
    assert hasattr(results, "statistics"), "Missing statistics"
    assert hasattr(results, "contexts"), "Missing contexts"

    # Check storage-specific attributes
    if storage_mode == "disk":
        assert hasattr(results, "output_files"), "Missing output_files for disk mode"
        assert results.output_files is not None, "output_files is None for disk mode"
        assert len(results.output_files) == len(results.experiment_config.configs), (
            "output_files length mismatch"
        )
        assert results.embeddings is None, "embeddings should be None for disk mode"
        print(f"✓ Disk mode: {len(results.output_files)} output files")
    else:  # memory or cpu
        assert hasattr(results, "embeddings"), "Missing embeddings for memory/cpu mode"
        assert results.embeddings is not None, "embeddings is None for memory/cpu mode"
        assert len(results.embeddings) == len(results.experiment_config.configs), (
            "embeddings length mismatch"
        )
        assert results.output_files is None, (
            "output_files should be None for memory/cpu mode"
        )

        # Check device placement
        if storage_mode == "cpu":
            # CPU mode: all embeddings should be on CPU
            for config_idx, config_embeddings in enumerate(results.embeddings):
                for doc_idx, emb in enumerate(config_embeddings):
                    assert emb.device.type == "cpu", (
                        f"Config {config_idx}, doc {doc_idx}: expected CPU device, got {emb.device}"
                    )
            print(f"✓ CPU mode: {len(results.embeddings)} configs in CPU memory")
        elif storage_mode == "memory":
            # Memory mode: embeddings should be on model device (GPU)
            expected_device = model_device
            for config_idx, config_embeddings in enumerate(results.embeddings):
                for doc_idx, emb in enumerate(config_embeddings):
                    assert emb.device.type == expected_device.split(":")[0], (
                        f"Config {config_idx}, doc {doc_idx}: expected {expected_device} device, got {emb.device}"
                    )
            print(
                f"✓ Memory mode: {len(results.embeddings)} configs in {expected_device} memory"
            )

    print("✓ Has all required attributes")

    # Check type
    assert isinstance(results, CompressionExperimentResults), (
        f"Wrong type: {type(results).__name__}"
    )
    print(f"✓ Correct type: {type(results).__name__}")

    # Check statistics structure
    stats = results.statistics
    required_stat_keys = [
        "num_documents",
        "num_configs",
        "config_token_counts",
        "avg_tokens_per_doc",
        "encoding_time",
        "compression_times",
        "total_time",
    ]
    for key in required_stat_keys:
        assert key in stats, f"Missing statistic: {key}"
    print(f"✓ Statistics complete with {len(stats)} fields")

    print("\n✓✓✓ Results structure test PASSED")


def test_statistics(results: CompressionExperimentResults, storage_mode: str):
    """Test statistics are valid and make sense."""
    print("\n" + "=" * 80)
    print(f"TEST: Statistics Validation (storage_mode={storage_mode})")
    print("=" * 80)

    stats = results.statistics

    # Print statistics
    print(f"\nDocuments encoded: {stats['num_documents']}")
    print(f"Configs tested: {stats['num_configs']}")
    print(f"Encoding time: {stats['encoding_time']:.3f}s")
    print(f"Total time: {stats['total_time']:.3f}s")
    print(f"Compression times: {[f'{t:.3f}s' for t in stats['compression_times']]}")

    # Validate counts
    assert stats["num_documents"] > 0, "No documents encoded"
    assert stats["num_configs"] == len(results.experiment_config.configs), (
        "Config count mismatch"
    )
    assert len(stats["config_token_counts"]) == stats["num_configs"], (
        "Token counts length mismatch"
    )
    assert len(stats["avg_tokens_per_doc"]) == stats["num_configs"], (
        "Avg tokens length mismatch"
    )
    print("✓ Counts are consistent")

    # Validate timing
    assert stats["encoding_time"] > 0, "No encoding time recorded"
    assert stats["total_time"] > stats["encoding_time"], (
        "Total time should exceed encoding time"
    )
    assert all(t >= 0 for t in stats["compression_times"]), "Negative compression time"
    print("✓ Timing values are valid")

    # Validate token counts
    for i, count in enumerate(stats["config_token_counts"]):
        assert count > 0, f"Config {i} has zero tokens"
    print("✓ All configs produced tokens")

    print("\n✓✓✓ Statistics validation test PASSED")


def test_compression_effectiveness(
    results: CompressionExperimentResults, storage_mode: str
):
    """Test that compression actually reduces tokens."""
    print("\n" + "=" * 80)
    print(f"TEST: Compression Effectiveness (storage_mode={storage_mode})")
    print("=" * 80)

    stats = results.statistics
    baseline_tokens = stats["config_token_counts"][0]

    print(f"\nBaseline: {baseline_tokens:,} tokens")

    # Check each compressed config
    for i in range(1, len(stats["config_token_counts"])):
        config = results.experiment_config.configs[i]
        config_name = results.experiment_config.config_names[i]
        compressed_tokens = stats["config_token_counts"][i]
        reduction_pct = (1 - compressed_tokens / baseline_tokens) * 100

        print(
            f"  [{i}] {config_name:30s}: {compressed_tokens:,} tokens ({reduction_pct:+.1f}% vs baseline)"
        )

        # Validate compression occurred
        if config and (config.pruning or config.pooling):
            assert compressed_tokens < baseline_tokens, (
                f"Config {i} ({config_name}) should reduce tokens but didn't"
            )

    print("\n✓ All compression configs reduced token count")
    print("\n✓✓✓ Compression effectiveness test PASSED")


def test_iteration_interface(results: CompressionExperimentResults, storage_mode: str):
    """Test iteration over results."""
    print("\n" + "=" * 80)
    print(f"TEST: Iteration Interface (storage_mode={storage_mode})")
    print("=" * 80)

    configs_iterated = 0
    for i, (config, embeddings) in enumerate(results):
        configs_iterated += 1
        desc = (
            config.description
            if config and hasattr(config, "description")
            else "Baseline"
        )

        # Validate embeddings
        assert isinstance(embeddings, list), f"Config {i} embeddings not a list"
        assert len(embeddings) > 0, f"Config {i} has no embeddings"
        assert all(isinstance(emb, torch.Tensor) for emb in embeddings), (
            f"Config {i} contains non-tensor embeddings"
        )

        print(f"  [{i}] {desc:30s}: {len(embeddings)} documents")

    # Check we iterated over all configs
    assert configs_iterated == len(results.experiment_config.configs), (
        "Didn't iterate over all configs"
    )

    print(f"\n✓ Successfully iterated over {configs_iterated} configs")
    print("\n✓✓✓ Iteration interface test PASSED")


def test_load_embeddings(results: CompressionExperimentResults, storage_mode: str):
    """Test loading embeddings by index."""
    print("\n" + "=" * 80)
    print(f"TEST: Load Embeddings (storage_mode={storage_mode})")
    print("=" * 80)

    for i in range(len(results.experiment_config.configs)):
        embeddings = results.load_embeddings(i)
        config_name = results.experiment_config.config_names[i]

        # Validate loaded embeddings
        assert isinstance(embeddings, list), f"Config {i} didn't return list"
        assert len(embeddings) > 0, f"Config {i} returned empty list"
        assert all(isinstance(emb, torch.Tensor) for emb in embeddings), (
            f"Config {i} contains non-tensors"
        )

        # Check token counts match statistics
        total_tokens = sum(len(emb) for emb in embeddings)
        expected_tokens = results.statistics["config_token_counts"][i]
        assert total_tokens == expected_tokens, (
            f"Config {i} token count mismatch: {total_tokens} != {expected_tokens}"
        )

        print(
            f"  [{i}] {config_name:30s}: loaded {len(embeddings)} docs, {total_tokens} tokens ✓"
        )

    print(
        f"\n✓ Successfully loaded embeddings for all {len(results.experiment_config.configs)} configs"
    )
    print("\n✓✓✓ Load embeddings test PASSED")


def test_disk_reload(results: CompressionExperimentResults, storage_mode: str):
    """Test that disk mode files can be reloaded correctly."""
    if storage_mode != "disk":
        return  # Skip for non-disk modes

    print("\n" + "=" * 80)
    print(f"TEST: Disk Mode Reload (storage_mode={storage_mode})")
    print("=" * 80)

    assert results.output_files is not None, "output_files should exist for disk mode"

    # Test reloading each config file directly
    for i in range(len(results.experiment_config.configs)):
        output_file = results.output_files[i]
        config_name = results.experiment_config.config_names[i]

        # Verify file exists
        assert output_file.exists(), f"Output file {output_file} does not exist"

        # Load file directly
        data = torch.load(output_file, map_location="cpu")
        assert "embeddings" in data, f"File {output_file} missing 'embeddings' key"
        assert "num_docs" in data, f"File {output_file} missing 'num_docs' key"

        loaded_embeddings = data["embeddings"]
        num_docs = data["num_docs"]

        # Validate loaded data
        assert isinstance(loaded_embeddings, list), f"Config {i} embeddings not a list"
        assert len(loaded_embeddings) == num_docs, (
            f"Config {i} doc count mismatch: {len(loaded_embeddings)} != {num_docs}"
        )
        assert all(isinstance(emb, torch.Tensor) for emb in loaded_embeddings), (
            f"Config {i} contains non-tensors"
        )

        # Check token counts match statistics
        total_tokens = sum(len(emb) for emb in loaded_embeddings)
        expected_tokens = results.statistics["config_token_counts"][i]
        assert total_tokens == expected_tokens, (
            f"Config {i} token count mismatch: {total_tokens} != {expected_tokens}"
        )

        # Verify embeddings are on CPU (as saved)
        for emb in loaded_embeddings:
            assert emb.device.type == "cpu", (
                f"Config {i}: embeddings should be on CPU, got {emb.device}"
            )

        print(
            f"  [{i}] {config_name:30s}: reloaded {len(loaded_embeddings)} docs, {total_tokens} tokens ✓"
        )

    # Test that load_embeddings() works correctly for disk mode
    print("\nTesting load_embeddings() method for disk mode...")
    for i in range(len(results.experiment_config.configs)):
        embeddings = results.load_embeddings(i, device="cpu")
        assert len(embeddings) > 0, f"Config {i} returned empty embeddings"
        assert all(emb.device.type == "cpu" for emb in embeddings), (
            f"Config {i}: load_embeddings() should return CPU tensors"
        )

        # Test loading to different device
        if torch.cuda.is_available():
            embeddings_gpu = results.load_embeddings(i, device="cuda")
            assert all(emb.device.type == "cuda" for emb in embeddings_gpu), (
                f"Config {i}: load_embeddings(device='cuda') should return CUDA tensors"
            )
            print(f"  [{i}] Successfully loaded to CPU and CUDA ✓")

    print(
        f"\n✓ Successfully reloaded all {len(results.experiment_config.configs)} configs from disk"
    )
    print("\n✓✓✓ Disk reload test PASSED")


def run_tests_for_storage_mode(
    model: ColBERT,
    documents: list[str],
    idf_stats: TokenTFIDFStats,
    storage_mode: str,
):
    """Run all tests for a specific storage mode."""
    print("\n" + "=" * 80)
    print(f"TESTING STORAGE MODE: {storage_mode.upper()}")
    print("=" * 80)

    # Get model device
    model_device = str(next(model.parameters()).device)

    # Create experiment config with specific storage mode
    experiment = CompressionExperimentConfig(
        configs=[
            None,  # Baseline
            CompressionConfig(
                description="IDF pruning k=10",
                pruning=[IDFPruningConfig(mode="document", top_k=10, stats=idf_stats)],
            ),
            CompressionConfig(
                description="IDF pruning k=20",
                pruning=[IDFPruningConfig(mode="document", top_k=20, stats=idf_stats)],
            ),
            CompressionConfig(
                description="Pooling f=2",
                pooling=[PoolingConfig(pool_factor=2)],
            ),
        ],
        output_dir=f"/tmp/test_experiment_integration_{storage_mode}",
        storage_mode=storage_mode,
        overwrite=True,
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    # Run experiment
    results = run_compression_experiment(
        model=model,
        documents=documents,
        experiment_config=experiment,
        batch_size=1024,
        show_progress=True,
    )

    # Run tests
    test_results_structure(results, storage_mode, model_device)
    test_statistics(results, storage_mode)
    test_compression_effectiveness(results, storage_mode)
    test_iteration_interface(results, storage_mode)
    test_load_embeddings(results, storage_mode)
    test_disk_reload(results, storage_mode)  # Only runs for disk mode

    return results


def run_all_tests():
    """Run all integration tests for all storage modes."""
    print("=" * 80)
    print("COMPRESSION EXPERIMENT INTEGRATION TESTS")
    print("Testing all storage modes: cpu, memory, disk")
    print("=" * 80)

    # Setup (done once, shared across all storage mode tests)
    model, documents, idf_stats = setup_model_and_data()

    # Test each storage mode
    storage_modes = ["cpu", "memory", "disk"]
    all_results = {}

    for storage_mode in storage_modes:
        try:
            results = run_tests_for_storage_mode(
                model=model,
                documents=documents,
                idf_stats=idf_stats,
                storage_mode=storage_mode,
            )
            all_results[storage_mode] = results
        except Exception as e:
            print(f"\n✗✗✗ TESTS FAILED FOR STORAGE MODE: {storage_mode} ✗✗✗")
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            raise

    # Final summary
    print("\n" + "=" * 80)
    print("✓✓✓ ALL TESTS PASSED FOR ALL STORAGE MODES ✓✓✓")
    print("=" * 80)
    print("\nTested storage modes:")
    for mode in storage_modes:
        print(f"  ✓ {mode.upper()}")
    print(
        "\nThe CompressionExperimentConfig is fully integrated into ColBERT.encode()!"
    )
    print("Encoding happens once, compression applied multiple times efficiently!")
    print("All storage modes (cpu, memory, disk) work correctly!")


if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n✗✗✗ TEST FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
