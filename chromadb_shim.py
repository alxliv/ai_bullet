"""Minimal compatibility shim for importing chromadb with NumPy 2.x fixes."""

from __future__ import annotations


def load_chromadb():
    """Import chromadb, adding NumPy aliases for deprecated names when needed."""
    try:
        import chromadb  # type: ignore
        return chromadb
    except AttributeError:
        import numpy as np  # type: ignore

        # Provide deprecated aliases expected by chromadb's older deps.
        if not hasattr(np, "float"):
            np.float = np.float64  # type: ignore[attr-defined]
        if not hasattr(np, "float_"):
            np.float_ = np.float64  # type: ignore[attr-defined]
        if not hasattr(np, "int"):
            np.int = np.int64  # type: ignore[attr-defined]
        if not hasattr(np, "int_"):
            np.int_ = np.int64  # type: ignore[attr-defined]
        if not hasattr(np, "complex"):
            np.complex = np.complex128  # type: ignore[attr-defined]

        import chromadb  # type: ignore
        return chromadb


chromadb = load_chromadb()

__all__ = ["chromadb", "load_chromadb"]
