"""
tokenizer_utils.py
==================

Unified tokenizer interface for the ai_bullet project.
Uses Hugging Face transformers AutoTokenizer for offline/local usage.

This module provides a consistent API for token encoding, decoding, counting,
and truncation across all components of the project.

Configuration:
--------------
Tokenizers are loaded from local directories specified in the TOKENIZER_MAP.
Run setup_tokenizers.py once while online to download and cache tokenizers.

For offline usage, tokenizers are loaded with local_files_only=True.
"""

import os
import time
import logging
from typing import List, Tuple, Optional
from functools import lru_cache
from config import USE_OPENAI, LLM_DEFAULT_MODEL

Verbose=True

logger = logging.getLogger(__name__)

if USE_OPENAI:
    # Try to import tiktoken for OpenAI models
    try:
        import tiktoken  # type: ignore
        HAS_TIKTOKEN = True
    except ImportError:
        HAS_TIKTOKEN = False
        logger.warning("tiktoken library not installed. Install with: pip install tiktoken")
else:
    # Try to import transformers library
    try:
        from transformers import AutoTokenizer
        HAS_TRANSFORMERS = True
    except ImportError:
        HAS_TRANSFORMERS = False
        logger.warning("transformers library not installed. Install with: pip install transformers")



# Map model names to local tokenizer directories
TOKENIZER_MAP = {
    "qwen3:4b-instruct-2507-fp16": "./tokenizers/qwen",
    "qwen3:4b-instruct": "./tokenizers/qwen",
    "qwen2.5": "./tokenizers/qwen",
    "qwen": "./tokenizers/qwen",
    "gpt-oss:20b": "./tokenizers/gpt-oss"
    # Add more models as needed
}

# Known OpenAI model encodings (fallback order matters)
OPENAI_MODEL_ENCODINGS = (
    ("gpt-4o-mini", "o200k_base"),
    ("gpt-4o", "o200k_base"),
    ("gpt-4.1", "o200k_base"),
    ("gpt-4-turbo", "o200k_base"),
    ("gpt-5", "o200k_base"),
    ("gpt-3.5", "cl100k_base"),
    ("text-embedding-3", "o200k_base"),
    ("text-embedding-ada-002", "cl100k_base"),
)


class _TiktokenWrapper:
    """Adapter to provide encode/decode like transformers tokenizers."""

    def __init__(self, encoding):
        self.encoding = encoding

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.encoding.decode(ids)


def _maybe_get_openai_tokenizer(model: str) -> Optional[_TiktokenWrapper]:
    model_lower = model.lower()
    for prefix, encoding_name in OPENAI_MODEL_ENCODINGS:
        if prefix in model_lower:
            if not HAS_TIKTOKEN:
                raise RuntimeError(
                    "tiktoken library not installed. Install with: pip install tiktoken"
                )
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding(encoding_name)
            logger.info(f"Using tiktoken encoding '{encoding.name}' for model '{model}'")
            return _TiktokenWrapper(encoding)
    return None

@lru_cache(maxsize=5)
def get_tokenizer(model: str = LLM_DEFAULT_MODEL):

    """
    Load and cache tokenizer for given model.

    Args:
        model: Model name to load tokenizer for

    Returns:
        AutoTokenizer instance

    Raises:
        RuntimeError: If transformers library is not available
        ValueError: If tokenizer path not found or doesn't exist
    """
    # Try OpenAI/tiktoken-backed encoder first
    openai_tokenizer = _maybe_get_openai_tokenizer(model)
    if openai_tokenizer is not None:
        return openai_tokenizer

    if not HAS_TRANSFORMERS:
        raise RuntimeError(
            "transformers library not installed. Install with: pip install transformers"
        )

    # Normalize model name
    model_key = model.lower()

    # Get tokenizer path from map
    tokenizer_path = TOKENIZER_MAP.get(model_key)

    if not tokenizer_path:
        raise ValueError(
            f"Model '{model}' not found in tokenizer map. "
            f"Available models: {', '.join(TOKENIZER_MAP.keys())}"
        )

    if not os.path.exists(tokenizer_path):
        raise ValueError(
            f"Tokenizer directory not found: {tokenizer_path}\n"
            f"Please run setup_tokenizers.py to download tokenizers first."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True  # Ensures offline operation
        )
        logger.info(f"Loaded tokenizer from: {tokenizer_path}")
        return tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}: {e}")


def encode(text: str, model: str = LLM_DEFAULT_MODEL) -> List[int]:
    """
    Encode text to token IDs.

    Args:
        text: Input text string
        model: Model name to use for tokenization

    Returns:
        List of token IDs

    Raises:
        RuntimeError: If tokenizer is not available
    """
    tokenizer = get_tokenizer(model)
    return tokenizer.encode(text)


def decode(ids: List[int], model: str = LLM_DEFAULT_MODEL) -> str:
    """
    Decode token IDs to text.

    Args:
        ids: List of token IDs
        model: Model name to use for tokenization

    Returns:
        Decoded text string

    Raises:
        RuntimeError: If tokenizer is not available
    """
    tokenizer = get_tokenizer(model)
    return tokenizer.decode(ids)


def count_tokens(text: str, model: str = LLM_DEFAULT_MODEL) -> int:
    """
    Count the number of tokens in text.

    Args:
        text: Input text string
        model: Model name to use for tokenization

    Returns:
        Number of tokens

    Raises:
        RuntimeError: If tokenizer is not available
    """
    tokenizer = get_tokenizer(model)
    return len(tokenizer.encode(text))


def truncate(text: str, max_tokens: int, model: str = LLM_DEFAULT_MODEL) -> Tuple[str, int]:
    """
    Truncate text to maximum token count.

    Args:
        text: Input text string
        max_tokens: Maximum number of tokens to keep
        model: Model name to use for tokenization

    Returns:
        Tuple of (truncated_text, actual_token_count)

    Raises:
        RuntimeError: If tokenizer is not available
    """
    tokenizer = get_tokenizer(model)
    tokens = tokenizer.encode(text)

    if len(tokens) <= max_tokens:
        return text, len(tokens)

    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)
    return truncated_text, len(truncated_tokens)


def split_by_tokens(text: str, max_tokens: int, model: str = LLM_DEFAULT_MODEL) -> List[str]:
    """
    Split text into chunks by token count.

    Args:
        text: Input text string
        max_tokens: Maximum tokens per chunk
        model: Model name to use for tokenization

    Returns:
        List of text chunks

    Raises:
        RuntimeError: If tokenizer is not available
    """
    tokenizer = get_tokenizer(model)

    start = time.perf_counter()
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))

    elapsed_ms = (time.perf_counter() - start) * 1000

    if Verbose:
        print(f"[{elapsed_ms:.2f} ms] split_by_tokens(text={len(text)} chars) of {model}, produced {len(tokens)} tokens and {len(chunks)} chunks")

    return chunks


# Convenience function for backward compatibility with tiktoken-style usage
def get_encoding(encoding_name: str = "cl100k_base"):
    """
    Dummy function for backward compatibility with tiktoken.
    Returns a mock object with encode method.

    Note: encoding_name is ignored as we always use the configured tokenizer.
    """
    class MockEncoding:
        def encode(self, text: str) -> List[int]:
            return encode(text)

        def decode(self, ids: List[int]) -> str:
            return decode(ids)

    return MockEncoding()


if __name__ == "__main__":
    # Test the tokenizer
    print("Testing tokenizer...")

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

    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use the tokenizer, please:")
        print("1. Install: pip install transformers")
        print("2. Run setup_tokenizers.py to download tokenizers")