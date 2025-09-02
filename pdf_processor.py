
"""Advanced PDF processing for lecture slide segmentation and deduplication.

This module implements an enhanced approach to dividing a lecture slides PDF
into logical sections. It combines semantic similarity (via embeddings),
visual similarity (via perceptual hashing) and title changes to determine
where major transitions occur. It also detects nearly identical pages to avoid
redundant summarisation.

If pdf2image or its underlying poppler dependency is not available, visual
features are skipped gracefully.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import PyPDF2
import fitz  # PyMuPDF

# Optional image-based features
try:
    from pdf2image import convert_from_path
    from PIL import Image
    import imagehash
    import pytesseract  # requires Tesseract installed
    _PDF2IMAGE_AVAILABLE = True
except Exception:  # pragma: no cover
    _PDF2IMAGE_AVAILABLE = False
    Image = None  # type: ignore
    imagehash = None  # type: ignore
    pytesseract = None  # type: ignore

from .embedding_utils import compute_embeddings, embed_pages_text

@dataclass
class Section:
    """Represents a contiguous group of pages belonging to one topic."""
    id: int
    title: str
    pages: List[int]

# ----------------------- Extraction helpers -----------------------

def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """Extract text from each page of a PDF using PyPDF2."""
    texts: List[str] = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num in range(len(reader.pages)):
            try:
                page = reader.pages[page_num]
                text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - corrupt pages
                logging.warning("Failed to extract text from page %s: %s", page_num, exc)
                text = ""
            texts.append(text)
    return texts

def extract_page_text_with_layout(pdf_path: str, page_idx: int) -> Dict[str, object]:
    """Extract rich text information for a single page using PyMuPDF.

    Returns a dictionary containing ``char_count``, ``block_count``,
    ``title_candidates`` and ``body_text``.
    """
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_idx)
        page_height = page.rect.height
        raw = page.get_text("dict")
        blocks = raw.get("blocks", [])

    texts: List[str] = []
    title_candidates: List[str] = []
    for b in blocks:
        if b.get("type") != 0:  # not text
            continue
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            if not line_text.strip():
                continue
            size = spans[0].get("size", 0) if spans else 0
            y0 = line.get("bbox", [0, 0, 0, 0])[1]
            texts.append(line_text)
            # crude heuristic: top quarter of page or large font
            if y0 < page_height * 0.25 or size > 16:
                title_candidates.append(line_text.strip())

    body_text = "\n".join(texts)
    return {
        "char_count": len(body_text),
        "block_count": len(blocks),
        "title_candidates": title_candidates,
        "body_text": body_text,
    }

def conditional_ocr_page_image(
    pdf_path: str,
    page_idx: int,
    meta: Dict[str, object],
    *,
    dpi: int = 400,
    lang: str = "kor+eng",
    ocr_threshold: int = 300,
) -> Dict[str, object]:
    """Augment ``meta`` with OCR text if insufficient characters were extracted."""
    if meta.get("char_count", 0) >= ocr_threshold:
        return meta
    if not _PDF2IMAGE_AVAILABLE:  # missing pdf2image or tesseract
        return meta
    try:
        image = convert_from_path(
            pdf_path, dpi=dpi, first_page=page_idx + 1, last_page=page_idx + 1
        )[0]
    except Exception:  # pragma: no cover - conversion failure
        return meta

    try:
        ocr_text = pytesseract.image_to_string(image, lang=lang) if pytesseract else ""
    except Exception:  # pragma: no cover - tesseract runtime failure
        ocr_text = ""

    combined = str(meta.get("body_text", "")).strip() + "\n" + ocr_text.strip()
    meta["body_text"] = combined.strip()
    meta["char_count"] = len(meta["body_text"])
    return meta

# ----------------------- Segmentation -----------------------

def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

def infer_section_title(page_meta_list: List[Dict[str, object]]) -> str:
    """Infer a representative title from metadata of pages in a section."""
    candidates: List[str] = []
    for meta in page_meta_list:
        candidates.extend(meta.get("title_candidates", []))
    if not candidates:
        text = page_meta_list[0].get("body_text", "")
        return text.strip().splitlines()[0][:50] if text else "Untitled Section"

    # Use rapidfuzz to group similar titles.
    from rapidfuzz import process as rf_process
    match = rf_process.extractOne(candidates[0], candidates)
    if match:
        best, score, _ = match
        if best and score > 60:
            return best
    return candidates[0]

def segment_into_sections(
    pdf_path: str,
    page_metas: List[Dict[str, object]],
    api_key: str,
    *,
    semantic_weight: float = 0.4,
    visual_weight: float = 0.2,
    title_weight: float = 0.4,
    duplicate_text_threshold: float = 0.98,
    duplicate_phash_threshold: int = 8,
    embedding_model: str = "text-embedding-3-small",
) -> Tuple[List[Section], Dict[int, int]]:
    """Compute sections and duplicates using a hybrid scoring scheme."""
    texts = [m.get("body_text", "") for m in page_metas]
    n_pages = len(texts)
    if n_pages == 0:
        return [], {}

    embeddings = embed_pages_text(texts, model=embedding_model, api_key=api_key)

    # Optional images & perceptual hashes
    images: Optional[List["Image.Image"]] = None
    try:
        if _PDF2IMAGE_AVAILABLE:
            images = convert_from_path(pdf_path, dpi=150)
    except Exception as exc:  # pragma: no cover
        logging.warning("Failed to convert PDF to images: %s", exc)

    if images is not None and imagehash:
        phashes = [_safe_phash(img) for img in images]
    else:
        phashes = [None] * n_pages  # type: ignore

    # Difference scores
    diff_scores: List[float] = []
    titles = [
        (m.get("title_candidates") or [""])[0].lower() if m.get("title_candidates") else ""
        for m in page_metas
    ]
    for i in range(n_pages - 1):
        sim_text = _cosine_similarity(embeddings[i], embeddings[i + 1])
        semantic_change = 1.0 - sim_text
        score = semantic_weight * semantic_change

        if phashes[i] is not None and phashes[i + 1] is not None:
            try:
                # normalised Hamming distance
                hamming = (phashes[i] - phashes[i + 1]) / (len(phashes[i].hash) ** 2)  # type: ignore
            except Exception:
                hamming = 0.0
            score += visual_weight * hamming

        if titles[i] and titles[i + 1] and titles[i] != titles[i + 1]:
            score += title_weight * 1.0

        diff_scores.append(score)

    threshold = float(np.mean(diff_scores) + np.std(diff_scores) * 0.5) if diff_scores else 0.0

    sections: List[Section] = []
    start = 0
    sec_id = 0
    for i, score in enumerate(diff_scores):
        if score > threshold:
            pages = list(range(start, i + 1))
            title = infer_section_title([page_metas[p] for p in pages])
            sections.append(Section(id=sec_id, title=title, pages=pages))
            sec_id += 1
            start = i + 1

    pages = list(range(start, n_pages))
    if pages:
        title = infer_section_title([page_metas[p] for p in pages])
        sections.append(Section(id=sec_id, title=title, pages=pages))

    # Duplicate detection
    duplicates: Dict[int, int] = {}
    for i in range(n_pages):
        for j in range(i + 1, n_pages):
            if j in duplicates:
                continue
            sim_text = _cosine_similarity(embeddings[i], embeddings[j])
            phash_sim = None
            if phashes[i] is not None and phashes[j] is not None:
                try:
                    phash_sim = (phashes[i] - phashes[j])  # type: ignore
                except Exception:
                    phash_sim = None
            if (sim_text >= duplicate_text_threshold) or (
                phash_sim is not None and phash_sim <= duplicate_phash_threshold
            ):
                duplicates[j] = i

    return sections, duplicates

def _safe_phash(img: "Image.Image"):
    try:
        return imagehash.phash(img.convert("RGB"))  # type: ignore
    except Exception as exc:  # pragma: no cover
        logging.warning("Failed to compute pHash: %s", exc)
        return None
