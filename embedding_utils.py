
"""Utilities for computing embeddings via the OpenAI API."""
from __future__ import annotations

import logging
import os
from typing import List, Iterable

import numpy as np
from openai import OpenAI

def compute_embeddings(
    texts: List[str],
    api_key: str,
    *,
    model: str = "text-embedding-3-small",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute embeddings for a list of texts using the OpenAI API."""
    client = OpenAI(api_key=api_key)
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = client.embeddings.create(model=model, input=batch)
        except Exception as exc:  # pragma: no cover - network errors
            logging.error(
                "Error calling OpenAI Embedding API for batch starting at %s: %s",
                i, exc,
            )
            # fallback: fill with zeros
            dim = len(embeddings[0]) if embeddings else 1536
            embeddings.extend([[0.0] * dim for _ in batch])
            continue
        sorted_items = sorted(response.data, key=lambda x: x.index)
        for item in sorted_items:
            embeddings.append(list(item.embedding))
    return np.array(embeddings)

def embed_texts(
    texts: List[str],
    *,
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Wrapper that reads the OpenAI API key from the environment if needed."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    return compute_embeddings(texts, api_key=key, model=model, batch_size=batch_size)

def embed_pages_text(
    pages_text: Iterable[str],
    *,
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
    batch_size: int = 32,
    max_retries: int = 3,
) -> np.ndarray:
    """Embed a sequence of page texts with basic retry logic."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    texts = list(pages_text)
    logging.info("Embedding %d pages; this may incur API costs", len(texts))
    client = OpenAI(api_key=key)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(model=model, input=batch)
                sorted_items = sorted(resp.data, key=lambda x: x.index)
                for item in sorted_items:
                    vectors.append(list(item.embedding))
                break
            except Exception as exc:  # pragma: no cover
                attempt += 1
                logging.warning("Embedding batch starting %s failed: %s", i, exc)
                if attempt >= max_retries:
                    dim = len(vectors[0]) if vectors else 1536
                    vectors.extend([[0.0] * dim for _ in batch])
                    break
    return np.array(vectors)

def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return a cosine similarity matrix for the given embeddings."""
    if embeddings.size == 0:
        return np.zeros((0, 0))
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normalised = embeddings / norm
    return normalised @ normalised.T
