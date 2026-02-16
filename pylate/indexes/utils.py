"""Utility functions for index operations."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

def get_process_memory_bytes() -> int:
    """Get current process resident set size (memory usage) in bytes using psutil.

    Returns
    -------
    int
        Memory usage in bytes, or -1 if unable to determine.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    except Exception:
        return -1


def format_memory(num_bytes: int) -> str:
    """Format a number of bytes into a human-readable string (B, KB, MB, GB, TB)."""
    if num_bytes < 0:
        return "unknown"
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    num = float(num_bytes)
    while num >= 1024 and i < len(suffixes) - 1:
        num /= 1024
        i += 1
    return f"{num:.2f} {suffixes[i]}"


def log_memory(label: str, verbose: bool = True) -> int:
    """Log current process memory usage with a label.

    Parameters
    ----------
    label
        A descriptive label for this memory checkpoint.
    verbose
        Whether to actually log (if False, still returns memory but doesn't log).

    Returns
    -------
    int
        Memory usage in bytes.
    """
    mem_bytes = get_process_memory_bytes()
    formatted = format_memory(mem_bytes)
    if verbose and mem_bytes >= 0:
        logger.info(f"[Memory] {label}: {formatted}")
    return mem_bytes

