"""Utilities to save generated explanations to Google Drive.

The previous version wrote a plain text file containing per‑page
summaries.  This revision formats a richer document that begins with a
title, date, original PDF filename and a table of contents listing page
ranges for each section.  Section headers are marked in bold markdown
(`**header**`) and each slide explanation is prefixed with ``페이지 N:``.

The function still targets the Colab/Google Drive workflow and keeps a
minimal dependency surface so it can run in the restricted environment
provided by the tests.  The Google Drive must be mounted by the caller
before invoking :func:`save_document_to_drive`.

"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, List, Tuple

try:  # Colab-specific import – ignored during unit tests
    from google.colab import drive  # type: ignore
except Exception:  # pragma: no cover
    drive = None  # type: ignore

SectionData = Tuple[str, List[Tuple[int, str]]]


def save_document_to_drive(
    title: str,
    pdf_filename: str,
    sections: Iterable[SectionData],
    drive_dir: str = "/content/drive/MyDrive",
) -> str:
    """Write the fully rendered document to a text file in Drive.

    Parameters
    ----------
    title:
        Document title.
    pdf_filename:
        Original PDF file name to record in the document header.
    sections:
        Iterable of ``(section_title, [(page_number, text), ...])``.
    drive_dir:
        Target directory inside the mounted Google Drive.
    """

    if drive is None:
        raise RuntimeError("Google Colab environment is required to use Drive.")

    # ``drive.mount`` is intentionally not called here because this function may be
    # executed from a non-interactive Python process (e.g. ``python -m``) where the
    # Colab helper cannot prompt for authentication.  The caller is expected to
    # mount the drive in an interactive cell beforehand.

    os.makedirs(drive_dir, exist_ok=True)

    file_path = os.path.join(drive_dir, f"{title}.txt")

    now = datetime.now().strftime("%Y-%m-%d")

    # Table of contents
    toc_lines: List[str] = ["섹션 목차:"]
    for section_title, slides in sections:
        if not slides:
            continue
        first, last = slides[0][0], slides[-1][0]
        toc_lines.append(f"- {section_title} (페이지 {first}–{last})")

    # Build document body
    lines: List[str] = [
        title,
        f"날짜: {now}",
        f"원본 파일명: {pdf_filename}",
        "",
        *toc_lines,
        "",
    ]

    for section_title, slides in sections:
        lines.append(f"**{section_title}**")
        for page_num, text in slides:
            lines.append(f"페이지 {page_num}:")
            lines.append(text.strip())
            lines.append("")
        lines.append("")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return file_path

