"""Utilities for computing embeddings via the OpenAI API.

The functions in this module encapsulate calls to the OpenAI Embedding
endpoint and return NumPy arrays. They are separated out so that
embedding computation can be reused across different processing steps.
"""

from __future__ import annotations

import logging
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
    """Compute embeddings for a list of texts using the OpenAI API.

    Parameters
    ----------
    texts : List[str]
        The list of strings to embed.
    api_key : str
        Your OpenAI API key.
    model : str, optional
        The embedding model to use, by default ``"text-embedding-3-small"``.
    batch_size : int, optional
        Number of texts to send per API request. Adjust this according
        to your rate limit and token quotas.

    Returns
    -------
    np.ndarray
        A 2D array of shape ``(len(texts), embedding_dim)`` where
        ``embedding_dim`` depends on the chosen model.
    """
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
        # The 'data' field is a list of dicts with 'embedding' and 'index'
        # sorted by index; we need to order them by index to align with batch
        # input order.
        sorted_items = sorted(response.get("data", []), key=lambda x: x["index"])
        for item in sorted_items:
            embeddings.append(item.get("embedding", []))
    return np.array(embeddings)