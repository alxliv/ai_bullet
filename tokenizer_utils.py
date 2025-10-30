"""
tokenizer_utils.py
==================

Unified tokenizer interface for the ai_bullet project.
Replaces OpenAI's tiktoken with Qwen3 tokenizer for offline/local usage.

This module provides a consistent API for token encoding, decoding, counting,
and truncation across all components of the project.

Configuration:
--------------
Set the TOKENIZER_PATH environment variable to point to your Qwen3 tokenizer.json file.
Example: TOKENIZER_PATH=/path/to/Qwen3-4B-Instruct/tokenizer.json

If not set, it will look for tokenizer.json in common locations:
- ./models/qwen3/tokenizer.json
- ./tokenizer.json
- ~/.cache/huggingface/hub/.../tokenizer.json
"""

import os
import logging
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import tokenizers library
try:
    from tokenizers import Tokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False
    logger.warning("tokenizers library not installed. Install with: pip install tokenizers")

# Global tokenizer instance
_tokenizer: Optional[Tokenizer] = None
_tokenizer_loaded = False


def _find_tokenizer_path() -> Optional[str]:
    """Search for tokenizer.json in common locations."""
    # Check environment variable first
    env_path = os.getenv("TOKENIZER_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Common locations to search
    search_paths = [
        "./models/qwen3/tokenizer.json",
        "./models/tokenizer.json",
        "./tokenizer.json",
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-4B-Instruct/snapshots/*/tokenizer.json"),
    ]

    for pattern in search_paths:
        if "*" in pattern:
            # Handle glob patterns
            from glob import glob
            matches = glob(pattern)
            if matches:
                return matches[0]
        elif os.path.isfile(pattern):
            return pattern

    return None


def _load_tokenizer() -> Optional[Tokenizer]:
    """Load the Qwen3 tokenizer from disk."""
    global _tokenizer, _tokenizer_loaded

    if _tokenizer_loaded:
        return _tokenizer

    _tokenizer_loaded = True

    if not HAS_TOKENIZERS:
        logger.error("Cannot load tokenizer: 'tokenizers' library not installed")
        return None

    tokenizer_path = _find_tokenizer_path()

    if not tokenizer_path:
        logger.error(
            "Tokenizer file not found. Please set TOKENIZER_PATH environment variable "
            "or place tokenizer.json in one of the expected locations."
        )
        return None

    try:
        _tokenizer = Tokenizer.from_file(tokenizer_path)
        logger.info(f"Loaded Qwen3 tokenizer from: {tokenizer_path}")
        return _tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        return None


def get_tokenizer() -> Optional[Tokenizer]:
    """Get the global tokenizer instance."""
    if _tokenizer is None and not _tokenizer_loaded:
        return _load_tokenizer()
    return _tokenizer


def encode(text: str) -> List[int]:
    """
    Encode text to token IDs.

    Args:
        text: Input text string

    Returns:
        List of token IDs

    Raises:
        RuntimeError: If tokenizer is not available
    """
    tok = get_tokenizer()
    if tok is None:
        raise RuntimeError("Tokenizer not available. Cannot encode text.")
    return tok.encode(text).ids


def decode(ids: List[int]) -> str:
    """
    Decode token IDs to text.

    Args:
        ids: List of token IDs

    Returns:
        Decoded text string

    Raises:
        RuntimeError: If tokenizer is not available
    """
    tok = get_tokenizer()
    if tok is None:
        raise RuntimeError("Tokenizer not available. Cannot decode tokens.")
    return tok.decode(ids)


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in text.

    Args:
        text: Input text string

    Returns:
        Number of tokens

    Raises:
        RuntimeError: If tokenizer is not available
    """
    tok = get_tokenizer()
    if tok is None:
        # Fallback to character-based estimation (4 chars per token)
        logger.warning("Tokenizer not available, using character-based estimation")
        return len(text) // 4
    return len(tok.encode(text).ids)


def truncate(text: str, max_tokens: int) -> Tuple[str, int]:
    """
    Truncate text to maximum token count.

    Args:
        text: Input text string
        max_tokens: Maximum number of tokens to keep

    Returns:
        Tuple of (truncated_text, actual_token_count)

    Raises:
        RuntimeError: If tokenizer is not available
    """
    tok = get_tokenizer()
    if tok is None:
        # Fallback to character-based truncation
        logger.warning("Tokenizer not available, using character-based truncation")
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text, len(text) // 4
        return text[:max_chars], max_tokens

    encoding = tok.encode(text)
    if len(encoding.ids) <= max_tokens:
        return text, len(encoding.ids)

    cut_ids = encoding.ids[:max_tokens]
    truncated_text = tok.decode(cut_ids)
    return truncated_text, len(cut_ids)


def split_by_tokens(text: str, max_tokens: int) -> List[str]:
    """
    Split text into chunks by token count.

    Args:
        text: Input text string
        max_tokens: Maximum tokens per chunk

    Returns:
        List of text chunks

    Raises:
        RuntimeError: If tokenizer is not available
    """
    tok = get_tokenizer()
    if tok is None:
        # Fallback to character-based splitting
        logger.warning("Tokenizer not available, using character-based splitting")
        max_chars = max_tokens * 4
        return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

    token_ids = tok.encode(text).ids
    chunks = []

    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i:i+max_tokens]
        chunk_text = tok.decode(chunk_ids)
        chunks.append(chunk_text)

    return chunks


# Convenience function for backward compatibility with tiktoken-style usage
def get_encoding(encoding_name: str = "cl100k_base"):
    """
    Dummy function for backward compatibility with tiktoken.
    Returns a mock object with encode method.

    Note: encoding_name is ignored as we always use Qwen3 tokenizer.
    """
    class MockEncoding:
        def encode(self, text: str) -> List[int]:
            return encode(text)

        def decode(self, ids: List[int]) -> str:
            return decode(ids)

    return MockEncoding()


if __name__ == "__main__":
    # Test the tokenizer
    print("Testing Qwen3 tokenizer...")

    test_text = "Hello, how are you? This is a test of the Qwen3 tokenizer."

    try:
        # Test encoding
        tokens = encode(test_text)
        print(f"Encoded tokens: {tokens}")
        print(f"Token count: {len(tokens)}")

        # Test decoding
        decoded = decode(tokens)
        print(f"Decoded text: {decoded}")

        # Test count_tokens
        count = count_tokens(test_text)
        print(f"Token count (direct): {count}")

        # Test truncate
        truncated, actual_count = truncate(test_text, 10)
        print(f"Truncated (10 tokens): {truncated}")
        print(f"Actual count: {actual_count}")

        # Test split_by_tokens
        chunks = split_by_tokens(test_text * 3, 20)
        print(f"Split into {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {chunk[:50]}...")

        print("\nTokenizer test successful!")

    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nTo use the tokenizer, please:")
        print("1. Install: pip install tokenizers")
        print("2. Set TOKENIZER_PATH environment variable or place tokenizer.json in expected location")
