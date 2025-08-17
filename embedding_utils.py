"""Utilities for computing embeddings via the OpenAI API.

This module wraps calls to the OpenAI embedding endpoint and exposes
helpers that return NumPy arrays for reuse across the application.
"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np
import openai


def compute_embeddings(
    texts: List[str],
    api_key: str,
    *,
    model: str = "text-embedding-3-small",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute embeddings for a list of texts using the OpenAI API."""
    openai.api_key = api_key
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = openai.Embedding.create(input=batch, model=model)
        except Exception as exc:
            logging.error("Error calling OpenAI Embedding API for batch starting at %s: %s", i, exc)
            # In case of failure, fill with zeros of appropriate dimension
            if embeddings:
                dim = len(embeddings[0])
            else:
                dim = 1536  # default dimension for most embedding models
            embeddings.extend([[0.0] * dim for _ in batch])
            continue
        sorted_items = sorted(response.get("data", []), key=lambda x: x["index"])
        for item in sorted_items:
            embeddings.append(item.get("embedding", []))
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
