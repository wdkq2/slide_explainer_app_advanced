"""Build the final text document for slide explanations."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Tuple

SectionData = Tuple[str, List[Tuple[int, str]]]


def build_document(
    title: str,
    pdf_filename: str,
    sections: Iterable[SectionData],
) -> str:
    """Return a formatted text document for the generated explanations.

    Parameters
    ----------
    title:
        Document title.
    pdf_filename:
        Original PDF file name to record in the document header.
    sections:
        Iterable of ``(section_title, [(page_number, text), ...])``.
    """

    now = datetime.now().strftime("%Y-%m-%d")

    toc_lines: List[str] = ["섹션 목차:"]
    for section_title, slides in sections:
        if not slides:
            continue
        first, last = slides[0][0], slides[-1][0]
        toc_lines.append(f"- {section_title} (페이지 {first}–{last})")

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

    return "\n".join(lines)
