"""
path_utils.py
=============
OS-agnostic path encoding/decoding utilities for ChromaDB storage.

This module provides functions to:
1. Encode absolute file paths to portable variable-based paths (e.g., $DOCS$/subfolder/file.pdf)
2. Decode variable-based paths back to absolute paths based on current OS configuration

Supported variables:
- $DOCS$ - Documentation root directory
- $SRC$ - Source code root directory
- $EXAMPLES$ - Examples root directory

Example:
    # Encoding (when storing to DB)
    abs_path = "D:/Work22/bullet3/docs/manual.pdf"
    encoded = encode_path(abs_path)  # Returns "$DOCS$/manual.pdf"

    # Decoding (when retrieving from DB)
    encoded = "$DOCS$/manual.pdf"
    abs_path = decode_path(encoded)  # Returns "D:/Work22/bullet3/docs/manual.pdf" on Windows
"""

import os
from pathlib import Path, PurePosixPath
from typing import Optional
from config import DOCUMENTS_PATH, SOURCES_PATH, EXAMPLES_PATH

# Expand user paths and normalize
DOCS_ROOT = os.path.normpath(os.path.expanduser(DOCUMENTS_PATH))
SRC_ROOT = os.path.normpath(os.path.expanduser(SOURCES_PATH))
EXAMPLES_ROOT = os.path.normpath(os.path.expanduser(EXAMPLES_PATH))

# Path variable mapping for encoding (absolute -> variable)
PATH_VARIABLES = {
    DOCS_ROOT: "$DOCS$",
    SRC_ROOT: "$SRC$",
    EXAMPLES_ROOT: "$EXAMPLES$",
}

# Reverse mapping for decoding (variable -> absolute)
VARIABLE_TO_PATH = {
    "$DOCS$": DOCS_ROOT,
    "$SRC$": SRC_ROOT,
    "$EXAMPLES$": EXAMPLES_ROOT,
}


def encode_path(absolute_path: str) -> str:
    """
    Convert an absolute file path to a portable variable-based path.

    Args:
        absolute_path: Absolute file path (e.g., "D:/Work22/bullet3/docs/manual.pdf")

    Returns:
        Encoded path with variable prefix (e.g., "$DOCS$/manual.pdf")
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


def decode_path(encoded_path: str) -> str:
    """
    Convert a variable-based path back to an absolute file path for the current OS.

    Args:
        encoded_path: Encoded path with variable prefix (e.g., "$DOCS$/manual.pdf")

    Returns:
        Absolute file path for current OS (e.g., "D:/Work22/bullet3/docs/manual.pdf")
        If no variable prefix found, returns the original path unchanged.
    """
    # Check if path starts with any known variable
    for variable, root in VARIABLE_TO_PATH.items():
        if encoded_path.startswith(variable + "/"):
            # Extract relative part (after variable and /)
            rel_part = encoded_path[len(variable) + 1:]
            # Convert to OS-specific path separator
            rel_path = rel_part.replace('/', os.sep)
            # Join with root
            return os.path.join(root, rel_path)

    # No variable found - return as-is
    return encoded_path


def is_encoded_path(path: str) -> bool:
    """
    Check if a path uses variable-based encoding.

    Args:
        path: Path string to check

    Returns:
        True if path starts with a known variable ($DOCS$, $SRC$, $EXAMPLES$)
    """
    return any(path.startswith(var + "/") for var in VARIABLE_TO_PATH.keys())


def get_path_variable(absolute_path: str) -> Optional[str]:
    """
    Get the path variable that matches an absolute path.

    Args:
        absolute_path: Absolute file path

    Returns:
        Path variable (e.g., "$DOCS$") or None if no match
    """
    norm_path = os.path.normpath(absolute_path)

    sorted_roots = sorted(PATH_VARIABLES.items(), key=lambda x: len(x[0]), reverse=True)

    for root, variable in sorted_roots:
        try:
            common = os.path.commonpath([norm_path, root])
            if os.path.normpath(common) == os.path.normpath(root):
                return variable
        except ValueError:
            continue

    return None


# Convenience function for migration scripts
def migrate_metadata_paths(metadata: dict) -> dict:
    """
    Update file_path in metadata dict from absolute to encoded format.

    Args:
        metadata: Metadata dict containing 'file_path' key

    Returns:
        Updated metadata dict with encoded path
    """
    if "file_path" in metadata:
        metadata["file_path"] = encode_path(metadata["file_path"])
    return metadata


if __name__ == "__main__":
    # Simple test/demo
    print("Path Encoding/Decoding Demo")
    print("=" * 50)

    test_paths = [
        os.path.join(DOCS_ROOT, "manual.pdf"),
        os.path.join(SRC_ROOT, "btRigidBody.cpp"),
        os.path.join(EXAMPLES_ROOT, "HelloWorld", "main.cpp"),
        "/some/random/path.txt",
    ]

    for path in test_paths:
        encoded = encode_path(path)
        decoded = decode_path(encoded)
        match = "OK" if os.path.normpath(decoded) == os.path.normpath(path) else "FAIL"

        print(f"\nOriginal:  {path}")
        print(f"Encoded:   {encoded}")
        print(f"Decoded:   {decoded}")
        print(f"Match:     {match}")
