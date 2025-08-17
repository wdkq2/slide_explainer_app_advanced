"""Advanced PDF processing for lecture slide segmentation and deduplication.

This module implements an enhanced approach to dividing a lecture
slides PDF into logical sections. Instead of relying solely on
differences in text length between adjacent pages, it combines
semantic similarity (via embeddings), visual similarity (via
perceptual hashing) and title changes to determine where major
transitions occur. It also detects nearly identical pages to avoid
redundant summarisation.

To handle PDFs consisting mainly of images, this module attempts to
convert pages to images using pdf2image. If pdf2image or its
underlying poppler dependency is not available, visual similarity
features are skipped gracefully.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import PyPDF2

try:
    from pdf2image import convert_from_path
    from PIL import Image
    import imagehash
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    # If pdf2image or imagehash is unavailable, we will skip visual features
    _PDF2IMAGE_AVAILABLE = False

from .embedding_utils import compute_embeddings


def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """Extract text from each page of a PDF using PyPDF2.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    Returns
    -------
    List[str]
        A list of strings, one per page.
    """
    texts: List[str] = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num in range(len(reader.pages)):
            try:
                page = reader.pages[page_num]
                text = page.extract_text() or ""
            except Exception as exc:
                logging.warning("Failed to extract text from page %s: %s", page_num, exc)
                text = ""
            texts.append(text)
    return texts


def _extract_images(pdf_path: str) -> Optional[List[Image.Image]]:
    """Convert each page of a PDF to a PIL Image.

    Returns None if pdf2image is not available or conversion fails.
    """
    if not _PDF2IMAGE_AVAILABLE:
        return None
    try:
        images = convert_from_path(pdf_path, dpi=150)
        return images
    except Exception as exc:
        logging.warning("Failed to convert PDF to images: %s", exc)
        return None


def _compute_phashes(images: List[Image.Image]) -> List[Optional[imagehash.ImageHash]]:
    """Compute perceptual hashes for a list of images.

    If imagehash is not available, returns a list of None.
    """
    phashes: List[Optional[imagehash.ImageHash]] = []
    if not _PDF2IMAGE_AVAILABLE:
        return [None] * len(images)
    for img in images:
        try:
            ph = imagehash.phash(img.convert("RGB"))
        except Exception as exc:
            logging.warning("Failed to compute pHash: %s", exc)
            ph = None
        phashes.append(ph)
    return phashes


def _first_nonempty_line(text: str) -> str:
    """Return the first non-empty line of a text string.

    Strips whitespace and ignores completely blank lines. If no
    non-empty line exists, returns an empty string.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero norm.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def advanced_split_pdf(
    pdf_path: str,
    api_key: str,
    *,
    expected_groups: int = 2,
    target_ratio: Optional[float] = None,
    embedding_model: str = "text-embedding-3-small",
    embedding_batch_size: int = 32,
    text_similarity_weight: float = 0.5,
    visual_similarity_weight: float = 0.3,
    title_change_weight: float = 0.2,
    duplicate_text_threshold: float = 0.98,
    duplicate_phash_threshold: int = 5,
) -> Tuple[List[List[int]], List[str], Dict[int, int]]:
    """Segment a PDF into logical groups using semantic and visual features.

    Parameters
    ----------
    pdf_path : str
        Path to the input PDF.
    api_key : str
        OpenAI API key for computing embeddings.
    expected_groups : int, optional
        The number of groups to partition the slides into. For most
        lectures this will be 2 (e.g., two topics). Values <1 will
        raise a ValueError.
    target_ratio : Optional[float], optional
        Approximate proportion of the document the first group should
        occupy. For example, ``0.55`` would bias the algorithm to
        split around 55% of the pages into the first group. If None,
        equal division is assumed.
    embedding_model : str, optional
        The OpenAI model identifier for computing embeddings.
    embedding_batch_size : int, optional
        The batch size to use when calling the embedding API.
    text_similarity_weight : float, optional
        Weight assigned to semantic change when computing the
        segmentation score. Must be non-negative.
    visual_similarity_weight : float, optional
        Weight assigned to visual change when computing the
        segmentation score. Ignored if image processing is unavailable.
    title_change_weight : float, optional
        Weight assigned to title changes when computing the segmentation
        score.
    duplicate_text_threshold : float, optional
        Cosine similarity threshold above which two pages are treated as
        duplicates based on text content.
    duplicate_phash_threshold : int, optional
        Hamming distance threshold below which two pages are treated as
        duplicates based on image similarity.

    Returns
    -------
    Tuple[List[List[int]], List[str], Dict[int, int]]
        A tuple containing:
        * A list of groups, each a list of page indices.
        * The extracted text per page.
        * A mapping from duplicate page indices to their canonical
          indices (only present for pages considered duplicates).
    """
    if expected_groups < 1:
        raise ValueError("expected_groups must be at least 1")

    # 1. Extract text from the PDF
    texts = extract_text_from_pdf(pdf_path)
    n_pages = len(texts)
    if n_pages == 0:
        return [], texts, {}

    # 2. Compute embeddings for semantic similarity
    embeddings = compute_embeddings(
        texts,
        api_key,
        model=embedding_model,
        batch_size=embedding_batch_size,
    )

    # 3. Extract images and compute perceptual hashes if possible
    images: Optional[List[Image.Image]] = _extract_images(pdf_path)
    phashes: List[Optional[imagehash.ImageHash]]
    if images is not None:
        phashes = _compute_phashes(images)
    else:
        phashes = [None] * n_pages

    # 4. Compute difference scores between adjacent pages
    diff_scores: List[float] = []
    for i in range(n_pages - 1):
        # Semantic change (1 - cosine similarity)
        sim_text = _cosine_similarity(embeddings[i], embeddings[i + 1])
        semantic_change = 1.0 - sim_text
        score = text_similarity_weight * semantic_change

        # Visual change (normalized Hamming distance) if available
        if phashes[i] is not None and phashes[i + 1] is not None:
            try:
                hamming = (phashes[i] - phashes[i + 1]) / (len(phashes[i].hash) ** 2)
            except Exception:
                hamming = 0.0
            score += visual_similarity_weight * hamming

        # Title change
        title_i = _first_nonempty_line(texts[i]).lower()
        title_j = _first_nonempty_line(texts[i + 1]).lower()
        title_change = 1.0 if title_i and title_j and title_i != title_j else 0.0
        score += title_change_weight * title_change

        diff_scores.append(score)
    # Append a terminal difference encouraging end-of-document splitting
    diff_scores.append(max(diff_scores) if diff_scores else 0.0)

    # 5. Determine segmentation points for the given number of groups
    if expected_groups == 1:
        groups = [list(range(n_pages))]
    else:
        # Determine target split index using ratio or equal division
        if target_ratio is None:
            target_ratio = 1.0 / expected_groups
        target_k = int(round(n_pages * target_ratio))
        # Compute best split indices (expected_groups - 1 cut points)
        cut_points: List[int] = []
        remaining_groups = expected_groups
        start = 0
        # We'll greedily select cut points one by one
        for g in range(expected_groups - 1):
            # Evaluate candidate cut positions between start and n_pages - (remaining_groups - 1)
            best_idx = None
            best_score = float("-inf")
            for i in range(start, n_pages - (remaining_groups - 1)):
                # Penalty: encourage splits near the target ratio for the remaining pages
                # relative proportion expected for this cut
                group_size_so_far = i - start + 1
                remaining_pages = n_pages - start
                # expected size for this group as ratio of remaining pages
                expected_size = remaining_pages * target_ratio
                penalty = abs(group_size_so_far - expected_size) / max(1, remaining_pages) * 0.1
                score = diff_scores[i] - penalty
                if score > best_score:
                    best_score = score
                    best_idx = i
            cut_points.append(best_idx)
            # Update start and remaining groups
            start = best_idx + 1
            remaining_groups -= 1
        # Construct groups based on cut points
        groups = []
        prev = 0
        for cp in cut_points:
            groups.append(list(range(prev, cp + 1)))
            prev = cp + 1
        groups.append(list(range(prev, n_pages)))

    # 6. Identify duplicates based on high similarity
    duplicates: Dict[int, int] = {}
    for i in range(n_pages):
        for j in range(i + 1, n_pages):
            if j in duplicates:
                continue  # Already marked as duplicate
            # Text similarity
            sim_text = _cosine_similarity(embeddings[i], embeddings[j])
            phash_sim = None
            if phashes[i] is not None and phashes[j] is not None:
                try:
                    phash_sim = (phashes[i] - phashes[j])
                except Exception:
                    phash_sim = None
            if (sim_text >= duplicate_text_threshold) or (
                phash_sim is not None and phash_sim <= duplicate_phash_threshold
            ):
                duplicates[j] = i

    return groups, texts, duplicates
