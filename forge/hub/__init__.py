"""HuggingFace Hub integration for Forge.

This module provides functionality to download and work with datasets
from the HuggingFace Hub.

Usage:
    from forge.hub import download_dataset, parse_hf_url

    # Parse hf:// URLs
    repo_id = parse_hf_url("hf://lerobot/pusht")

    # Download a dataset
    local_path = download_dataset("lerobot/pusht")
"""

from forge.hub.download import (
    download_dataset,
    find_in_hf_cache,
    get_cache_dir,
    get_hf_cache_dir,
)
from forge.hub.url import is_hf_url, parse_hf_url

__all__ = [
    "download_dataset",
    "find_in_hf_cache",
    "get_cache_dir",
    "get_hf_cache_dir",
    "is_hf_url",
    "parse_hf_url",
]
