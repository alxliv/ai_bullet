"""
path_utils.py
=============
OS-agnostic path encoding/decoding utilities for ChromaDB storage.

This module provides functions to:
1. Encode absolute file paths to portable variable-based paths (e.g., {DOCS}/subfolder/file.pdf)
2. Decode variable-based paths back to absolute paths based on current OS configuration

Supported variables:
- {DOCS} - Documentation root directory
- {SRC} - Source code root directory
- {EXAMPLES} - Examples root directory

Example:
    # Encoding (when storing to DB)
    abs_path = "D:/Work22/bullet3/docs/manual.pdf"
    encoded = encode_path(abs_path)  # Returns "docs/manual.pdf"

"""

import os
from config import GLOBAL_RAGDATA_MAP

def _normalize_root(path: str) -> str:
    """Expand ~ and collapse separators for consistent comparisons."""
    return os.path.normpath(os.path.expanduser(path))

# Build variable mappings dynamically from GLOBAL_RAGDATA_MAP
PATH_VARIABLES = {
    _normalize_root(path): f"/{key.lower()}"
    for key, (path, _) in GLOBAL_RAGDATA_MAP.items()
}

def encode_path(absolute_path: str) -> str:
    """
    Convert an absolute file path to a portable variable-based path.

    Args:
        absolute_path: Absolute file path (e.g., "D:/Work22/bullet3/docs/manual.pdf")

    Returns:
        Encoded path with variable prefix (e.g., "{DOCS}/manual.pdf")
        If path doesn't match any known root, returns the original path unchanged.

    Note:
        - Uses forward slashes (/) as separator for OS-agnostic storage
        - Paths are case-sensitive on Unix, case-insensitive on Windows
    """
    # Normalize the input path
    norm_path = os.path.normpath(absolute_path)

    # Try to match against known roots (longest match first)
    sorted_roots = sorted(PATH_VARIABLES.items(), key=lambda x: len(x[0]), reverse=True)

    for root, variable in sorted_roots:
        # Check if path starts with this root
        # Use os.path.commonpath for reliable prefix matching
        try:
            common = os.path.commonpath([norm_path, root])
            if os.path.normpath(common) == os.path.normpath(root):
                # Extract relative part
                rel_path = os.path.relpath(norm_path, root)
                # Convert to forward slashes for storage (POSIX-style)
                posix_rel = rel_path.replace(os.sep, '/')
                return f"{variable}/{posix_rel}"
        except ValueError:
            # Paths are on different drives (Windows) - skip
            continue

    # No match found - return original path
    return absolute_path


