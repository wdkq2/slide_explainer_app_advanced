"""LLM interface for summarising lecture slide content.

This module provides a helper to call the OpenAI Chat API to produce
per‑page summaries of extracted slide text. Summaries are generated
separately for each group of pages to preserve intra‑section context
while avoiding repetition across sections.

To use this module you must supply a valid OpenAI API key. See
``README.md`` for details on configuring your environment.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from openai import OpenAI


def summarize_groups(
    api_key: str,
    texts: List[str],
    groups: List[List[int]],
    *,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
) -> Dict[int, str]:
    """Generate summaries for pages of a PDF grouped by context.

    Parameters
    ----------
    api_key : str
        Your OpenAI API key.
    texts : List[str]
        List of extracted text for each page in the document.
    groups : List[List[int]]
        A list of groups, where each group is a list of page indices.
        Pages within the same group will be summarised sequentially
        using conversational context to avoid repetition.
    model : str, optional
        OpenAI model name to use for summarisation, by default
        ``"gpt-3.5-turbo"``.
    temperature : float, optional
        Sampling temperature for the model. Lower values make the
        output more deterministic, by default ``0.3``.

    Returns
    -------
    Dict[int, str]
        A mapping from page index to the generated summary string.
    """
    client = OpenAI(api_key=api_key)
    summaries: Dict[int, str] = {}

    # System prompt instructing the model to summarise slides in Korean
    system_prompt = (
        "당신은 대학 전공 수업 슬라이드를 요약하는 한국어 도우미입니다. "
        "현재 슬라이드의 핵심 내용을 간결하게 설명하되, 이전 슬라이드에서 이미 설명한 내용을 반복하지 마세요. "
        "요약은 친절하지만 간략하게 작성해 주세요."
    )

    for group in groups:
        previous_summaries: List[str] = []
        for idx in group:
            page_text = texts[idx] or ""
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            if previous_summaries:
                context = "\n\n".join(
                    [f"페이지 {i + 1} 요약: {s}" for i, s in enumerate(previous_summaries)]
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "다음은 앞서 요약한 슬라이드 내용입니다. 새로운 내용과 중복을 피하세요.\n"
                            f"{context}"
                        ),
                    }
                )
            if page_text.strip():
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "현재 슬라이드의 본문 내용은 다음과 같습니다:\n"
                            f"{page_text}\n\n"
                            "위 내용을 바탕으로 이 슬라이드의 요약을 작성하세요."
                        ),
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "현재 슬라이드에서 추출 가능한 텍스트가 없습니다. "
                            "이미지나 표만 있는 슬라이드라면 해당 이미지나 표의 핵심 주제를 추론하여 짧게 요약해 주세요."
                        ),
                    }
                )
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                summary = response.choices[0].message.content.strip()
            except Exception as exc:
                logging.error(
                    "Error during OpenAI API call for page %s: %s", idx, exc
                )
                summary = "(요약을 생성하는 데 실패했습니다.)"
            summaries[idx] = summary
            previous_summaries.append(summary)
    return summaries


def explain_section(
    items: List[Tuple[int, str]],
    section_title: str,
    *,
    model: str = "gpt-4o",
    language: str = "ko",
    min_sentences_per_slide: int = 5,
    max_tokens: int = 2200,
    temperature: float = 0.2,
) -> str:
    """Return detailed explanations for slides within a section."""

    client = OpenAI()

    system_prompt = (
        "당신은 대학 3학년 공대 과목의 조교입니다. 처음 보는 학생에게도 "
        "맥락적으로 자세한 설명을 제공하세요. 한 줄 요약 금지." 
        f"각 슬라이드는 최소 {min_sentences_per_slide}문장으로 설명하세요."
    )

    user_lines = [f"섹션 개요: {section_title}"]
    for page, text in items:
        user_lines.append(f"페이지 {page}: {text}")
    user_content = "\n".join(user_lines)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                user_content
                + "\n\n슬라이드별로 '페이지 N:' 형식으로 자세히 설명하세요. "
                "이미 설명한 내용을 반복하지 말고 다음 슬라이드 내용을 선점하지 마세요."
            ),
        },
    ]

    def _call() -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    import re

    out = _call()
    for _ in range(1):
        ok = True
        for page, _ in items:
            if not re.search(rf"^페이지 {page}:", out, re.MULTILINE):
                ok = False
                break
        if ok:
            break
        out = _call()
    return out
