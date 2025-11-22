"""Unified embedding client supporting OpenAI and Ollama backends."""
from __future__ import annotations

import os
from typing import List, Optional
import httpx
import time

Verbose=True

from config import (
    EMBEDDING_MODEL,
    OPENAI_BASE_URL,
    OLLAMA_BASE_URL
)

from dotenv import load_dotenv
load_dotenv()

class EmbedClientUni:
    """Provide a single embed() API regardless of provider."""

    def __init__(
        self,
        *,
        use_openai: bool,
        timeout: Optional[float] = None,
    ) -> None:
        self.use_openai = use_openai
        self.embedding_model = EMBEDDING_MODEL
        self._timeout = timeout

        if use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required when USE_OPENAI is enabled")

            base_url = OPENAI_BASE_URL.rstrip("/")
            self._openai_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            self._client = httpx.Client(base_url=base_url, timeout=timeout)
            self._openai_model = EMBEDDING_MODEL
        else:
            base_url = OLLAMA_BASE_URL.rstrip("/")
            self._ollama_model = EMBEDDING_MODEL
            self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def embed(self, text: str) -> List[float]:
        start = time.perf_counter()
        if self.use_openai:
            payload = {"model": self._openai_model, "input": text}
            response = self._client.post(
                "/embeddings", headers=self._openai_headers, json=payload
            )
            response.raise_for_status()
            data = response.json()
            try:
                embedding = data["data"][0]["embedding"]
            except (KeyError, IndexError) as exc:
                raise RuntimeError("OpenAI embedding response missing data") from exc
            if not isinstance(embedding, list):
                raise RuntimeError("OpenAI embedding response is not a list of floats")
        else:
            payload = {"model": self._ollama_model, "prompt": text}
            response = self._client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("Ollama embeddings response missing 'embedding' vector")

        elapsed_ms = (time.perf_counter() - start) * 1000
        if Verbose:
            print(f"[{elapsed_ms:.2f} ms] embed(text={len(text)} chars) of {payload['model']}, produced {len(embedding)} embedding vector.")

        return embedding

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EmbedClientUni":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
