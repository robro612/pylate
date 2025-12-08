#!/usr/bin/env python3
"""
Merge multiple JSONL results files from compression experiments.

This script merges results from multiple experiment runs, carefully handling metadata
and ensuring consistency across merged files.

Usage:
    python merge_results.py <output_file> <input_file1> <input_file2> [...]
    python merge_results.py merged_results.jsonl results1.jsonl results2.jsonl results3.jsonl

The script will:
1. Validate that all input files have compatible metadata (same model, dataset, metrics)
2. Merge metadata from all runs
3. Combine all result entries
4. Write to a single output JSONL file
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from datetime import datetime
from collections import defaultdict
import math


def load_jsonl_file(filepath: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Load a JSONL results file and separate metadata from results.
    
    Parameters
    ----------
    filepath : Path
        Path to the JSONL file
        
    Returns
    -------
    tuple[dict | None, list[dict]]
        (metadata_entry, result_entries)
        metadata_entry is None if no metadata line found
    """
    metadata = None
    results = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in {filepath}: {e}", file=sys.stderr)
                continue
            
            entry_type = entry.get("type")
            if entry_type == "metadata":
                if metadata is not None:
                    print(f"Warning: Multiple metadata entries in {filepath}, using first one", file=sys.stderr)
                else:
                    metadata = entry
            elif entry_type == "result":
                results.append(entry)
            else:
                print(f"Warning: Unknown entry type '{entry_type}' at line {line_num} in {filepath}", file=sys.stderr)
    
    return metadata, results


def validate_compatibility(metadatas: list[dict[str, Any]], filepaths: list[Path]) -> None:
    """
    Validate that all metadata entries are compatible for merging.
    
    Parameters
    ----------
    metadatas : list[dict]
        List of metadata dictionaries
    filepaths : list[Path]
        Corresponding file paths (for error messages)
        
    Raises
    ------
    ValueError
        If metadata entries are incompatible
    """
    if not metadatas:
        print("Warning: No metadata entries found in any input file", file=sys.stderr)
        return
    
    # Check model name consistency
    model_names = [m.get("model_name") for m in metadatas if m.get("model_name")]
    if len(set(model_names)) > 1:
        print(f"Warning: Multiple model names found: {set(model_names)}", file=sys.stderr)
        print("  This may indicate incompatible experiments", file=sys.stderr)
    
    # Check dataset name consistency
    dataset_names = [m.get("dataset_name") for m in metadatas if m.get("dataset_name")]
    if len(set(dataset_names)) > 1:
        print(f"Warning: Multiple dataset names found: {set(dataset_names)}", file=sys.stderr)
        print("  This may indicate incompatible experiments", file=sys.stderr)
    
    # Check metrics consistency
    metrics_sets = [tuple(sorted(m.get("args", {}).get("metrics", []))) for m in metadatas]
    if len(set(metrics_sets)) > 1:
        print(f"Warning: Different metrics used across experiments", file=sys.stderr)
        print("  Some results may have missing metric values", file=sys.stderr)


def merge_metadata(metadatas: list[dict[str, Any]], filepaths: list[Path]) -> dict[str, Any]:
    """
    Merge metadata from multiple experiments into a single metadata entry.
    
    Parameters
    ----------
    metadatas : list[dict]
        List of metadata dictionaries from different runs
    filepaths : list[Path]
        Corresponding file paths
        
    Returns
    -------
    dict
        Merged metadata entry
    """
    if not metadatas:
        # Create minimal metadata if none exists
        return {
            "type": "metadata",
            "run_id": "merged_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "merged_from": [str(fp) for fp in filepaths],
            "num_source_files": len(filepaths),
        }
    
    # Use first metadata as base
    merged = metadatas[0].copy()
    
    # Update with merge-specific information
    merged["run_id"] = "merged_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    merged["timestamp"] = datetime.now().isoformat()
    merged["merged_from"] = [str(fp) for fp in filepaths]
    merged["num_source_files"] = len(filepaths)
    
    # Collect all source run IDs
    source_run_ids = [m.get("run_id") for m in metadatas if m.get("run_id")]
    if source_run_ids:
        merged["source_run_ids"] = source_run_ids
    
    # Aggregate timing information
    total_encoding_time = sum(m.get("timing", {}).get("encoding_time", 0) for m in metadatas)
    total_query_encoding_time = sum(m.get("timing", {}).get("query_encoding_time", 0) for m in metadatas)
    total_time = sum(m.get("timing", {}).get("total_time", 0) for m in metadatas)
    
    merged["timing"] = {
        "total_encoding_time": total_encoding_time,
        "total_query_encoding_time": total_query_encoding_time,
        "total_time": total_time,
    }

    # Aggregate config information
    all_configs = []
    for m in metadatas:
        configs = m.get("configs", [])
        all_configs.extend(configs)

    if all_configs:
        merged["configs"] = all_configs
        merged["num_configs"] = len(all_configs)

    return merged


def merge_results(
    result_lists: list[list[dict[str, Any]]],
    filepaths: list[Path],
    renumber_configs: bool = False,
) -> list[dict[str, Any]]:
    """
    Merge result entries from multiple experiments.

    Parameters
    ----------
    result_lists : list[list[dict]]
        List of result entry lists from different runs
    filepaths : list[Path]
        Corresponding file paths (for source tracking)
    renumber_configs : bool
        If True, renumber config_idx sequentially across all merged results

    Returns
    -------
    list[dict]
        Merged (with aggregation) and optionally renumbered result entries.
        When multiple runs share the same config, results are averaged and
        *_std fields are added to capture variability (error bars).
    """
    all_results = []

    for filepath, results in zip(filepaths, result_lists):
        for result in results:
            # Add source file information
            result_copy = result.copy()
            result_copy["source_file"] = str(filepath)
            result_copy["original_run_id"] = result.get("run_id")
            all_results.append(result_copy)

    # Group results by their config (serialized) so we can average duplicates
    grouped: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for res in all_results:
        cfg_key = json.dumps(res.get("config", None), sort_keys=True) if "config" in res else None
        grouped[cfg_key].append(res)

    aggregated_results: list[dict[str, Any]] = []

    def _mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return mean, math.sqrt(var)

    for _, group in grouped.items():
        if len(group) == 1:
            aggregated_results.append(group[0])
            continue

        base = group[0].copy()
        base["num_runs"] = len(group)
        base["source_files"] = [g.get("source_file") for g in group]
        base["original_run_ids"] = [g.get("original_run_id") for g in group]
        base["original_config_idxs"] = [g.get("config_idx") for g in group]
        base["runfile_paths"] = [g.get("runfile_path") for g in group if g.get("runfile_path")]

        # Average numeric summary fields
        for field in ["token_count", "avg_tokens_per_doc", "compression_time"]:
            vals = [g.get(field) for g in group if g.get(field) is not None]
            mean, std = _mean_std(vals)
            base[field] = mean
            base[field + "_std"] = std

        # Average metrics
        merged_eval: dict[str, float] = {}
        metric_keys = set()
        for g in group:
            metric_keys.update((g.get("evaluation") or {}).keys())

        for metric in metric_keys:
            vals = [
                (g.get("evaluation") or {}).get(metric)
                for g in group
                if g.get("evaluation") and (g.get("evaluation") or {}).get(metric) is not None
            ]
            mean, std = _mean_std(vals)
            merged_eval[metric] = mean
            merged_eval[metric + "_std"] = std

        base["evaluation"] = merged_eval
        aggregated_results.append(base)

    # Optionally renumber config indices after aggregation
    if renumber_configs:
        for new_idx, result in enumerate(aggregated_results):
            result["original_config_idx"] = result.get("config_idx")
            result["config_idx"] = new_idx

    # Stable-ish ordering: by config_name then config_idx
    aggregated_results.sort(key=lambda r: (r.get("config_name", ""), r.get("config_idx", 0)))

    return aggregated_results


def save_merged_jsonl(
    output_path: Path,
    metadata: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """
    Save merged metadata and results to a JSONL file.

    Parameters
    ----------
    output_path : Path
        Output file path
    metadata : dict
        Merged metadata entry
    results : list[dict]
        Merged result entries
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        # Write metadata as first line
        f.write(json.dumps(metadata, default=str) + '\n')

        # Write all results
        for result in results:
            f.write(json.dumps(result, default=str) + '\n')

    print(f"✓ Merged {len(results)} results from {metadata.get('num_source_files', 0)} files")
    print(f"✓ Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple JSONL results files from compression experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge multiple result files
  python merge_results.py merged.jsonl results1.jsonl results2.jsonl results3.jsonl

  # Merge with config renumbering
  python merge_results.py --renumber merged.jsonl results*.jsonl

  # Merge from a specific directory
  python merge_results.py merged.jsonl results/experiment_*/results_*.jsonl
        """
    )

    parser.add_argument(
        "output_file",
        type=str,
        help="Output JSONL file path"
    )

    parser.add_argument(
        "input_files",
        type=str,
        nargs="+",
        help="Input JSONL files to merge"
    )

    parser.add_argument(
        "--renumber",
        action="store_true",
        help="Renumber config_idx sequentially across all merged results"
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on metadata incompatibilities (default: warn only)"
    )

    args = parser.parse_args()

    # Convert paths
    output_path = Path(args.output_file)
    input_paths = [Path(f) for f in args.input_files]

    # Validate input files exist
    missing_files = [p for p in input_paths if not p.exists()]
    if missing_files:
        print(f"Error: The following input files do not exist:", file=sys.stderr)
        for p in missing_files:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(input_paths)} JSONL files...")
    print()

    # Load all files
    metadatas = []
    result_lists = []

    for filepath in input_paths:
        print(f"Loading: {filepath}")
        metadata, results = load_jsonl_file(filepath)
        metadatas.append(metadata)
        result_lists.append(results)
        print(f"  Found: {len(results)} results")
        if metadata:
            print(f"  Run ID: {metadata.get('run_id', 'N/A')}")
            print(f"  Model: {metadata.get('model_name', 'N/A')}")
            print(f"  Dataset: {metadata.get('dataset_name', 'N/A')}")
        print()

    # Filter out None metadatas for validation
    valid_metadatas = [m for m in metadatas if m is not None]

    # Validate compatibility
    print("Validating compatibility...")
    try:
        validate_compatibility(valid_metadatas, input_paths)
        print("✓ Validation complete")
    except ValueError as e:
        if args.strict:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"Warning: {e}", file=sys.stderr)
    print()

    # Merge metadata
    print("Merging metadata...")
    merged_metadata = merge_metadata(valid_metadatas, input_paths)
    print(f"✓ Created merged metadata with run_id: {merged_metadata.get('run_id')}")
    print()

    # Merge results
    print("Merging results...")
    merged_results = merge_results(result_lists, input_paths, renumber_configs=args.renumber)
    print(f"✓ Merged {len(merged_results)} total results")
    if args.renumber:
        print("✓ Renumbered config indices")
    print()

    # Save merged file
    print("Saving merged file...")
    save_merged_jsonl(output_path, merged_metadata, merged_results)
    print()
    print("Done!")


if __name__ == "__main__":
    main()
