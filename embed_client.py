"""Unified embedding client supporting OpenAI and Ollama backends."""
from __future__ import annotations

import os
from typing import List, Optional

import httpx


class EmbedClientUni:
    """Provide a single embed() API regardless of provider."""

    def __init__(
        self,
        *,
        use_openai: bool,
        embedding_model: str,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.use_openai = use_openai
        self.embedding_model = embedding_model
        self._timeout = timeout

        if use_openai:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required when USE_OPENAI is enabled")

            base_url = (
                openai_base_url
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            ).rstrip("/")
            self._openai_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            self._client = httpx.Client(base_url=base_url, timeout=timeout)
            self._openai_model = embedding_model
        else:
            base_url = (
                ollama_base_url
                or os.getenv("OLLAMA_BASE_URL")
                or "http://127.0.0.1:11434"
            ).rstrip("/")
            self._ollama_model = (
                ollama_model
                or os.getenv("OLLAMA_EMBED_MODEL")
                or embedding_model
            )
            self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def embed(self, text: str) -> List[float]:
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
            return embedding

        payload = {"model": self._ollama_model, "prompt": text}
        response = self._client.post("/api/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Ollama embeddings response missing 'embedding' vector")
        return embedding

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EmbedClientUni":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
