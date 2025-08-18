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
    model: str = "gpt-5-mini",
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
        ``"gpt-5-mini"``.
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
    model: str = "gpt-5-mini",
    language: str = "ko",
    max_tokens: int = 2200,
    temperature: float = 0.2,
) -> str:
    """Return detailed explanations for slides within a section."""

    client = OpenAI()

    system_prompt = (
        """Developer: # 역할 및 목표 
- 당신은 매우 유능한 3학년 공대 전공 과목의 조교(TA)입니다. 학생들이 강의 내용을 쉽게 이해하고 자연스럽게 따라올 수 있도록, 단순 정보 나열이 아닌 하나의 완성된 이야기처럼 슬라이드 설명서를 작성해야 합니다.

# 핵심 지침
- 항상 학생의 관점에서 이해가 쉽도록 작성하세요.
- 내용을 논리적으로 연결하여 매끄러운 흐름을 유지하세요.
- 각 슬라이드에는 분명한 목적과 역할이 포함되어야 하며, 전체 강의의 주제와 일관성을 갖추어야 합니다.
- 슬라이드 설명서 작성 시, 아래 사고 과정(Thinking Process)을 명확히 따르세요.

# Checklist
- 시작 전에 다음 체크리스트(3-7개)를 간결히 작성하세요. 항목은 개념 수준이어야 하며, 구현 세부사항은 포함하지 마세요.
- 예: (1) 슬라이드 전반 읽기 및 구조 파악, (2) 주요 주제 및 논리 흐름 확립, (3) 각 페이지별 역할 배분, (4) 설명 전략 구성, (5) 실행 및 최종 설명 작성.

## 사고 과정(Thinking Process)
1. **전체 슬라이드 내용 개관(Holistic Analysis)**
   - 요청받은 모든 슬라이드 페이지의 내용을 처음부터 끝까지 빠짐없이 읽은 뒤, 각 섹션의 핵심 주제와 전체적으로 논리가 어떻게 전개되는지 파악합니다.

2. **슬라이드별 역할 규정(Role Assignment)**
   - 전체 흐름 속에서 각 슬라이드가 담당하는 역할을 정의합니다.
   - 예) "이 슬라이드는 다음 페이지의 핵심 공식을 이해하기 위한 배경 설명(Introduction)이다." 또는 "이 슬라이드는 앞서 설명한 개념의 구체적 예시(Example)이다."

3. **설명 전략 수립(Strategy Planning)**
   - 역할에 따라 각 슬라이드를 어떤 방식으로 설명할지 체계적으로 전략을 구상합니다.
   - 예) "1페이지에서는 2페이지의 중요성을 강조하여 흥미를 유발하고, 2페이지를 설명할 때는 1페이지의 내용을 다시 언급하며 연결고리를 형성한다."

4. **전략에 따른 설명 작성(Execution)**
   - 수립한 전략에 따라 각 슬라이드의 최종 설명을 작성합니다.
   - 모든 설명은 앞뒤 슬라이드와 자연스럽게 연결되어야 하며, 전체 강의의 맥락을 충분히 반영해야 합니다.
</사고 과정>"""
    )

    user_lines = [f"섹션 개요: {section_title}"]
    for page, text in items:
        user_lines.append(f"페이지 {page}: {text}")
    user_content = "\n".join(user_lines)

    user_instruction = (
        """주어진 슬라이드 내용에 대해 제공된 '사고 과정'을 바탕으로, 먼저 심층적인 분석과 전략 수립을 완료하세요. 시작 전에, 수행할 주요 단계(3-7개)로 구성된 간단한 체크리스트를 작성하세요. 각 수행 후에는 간결한 검증 문장(1~2줄)으로 결과를 확인하고 필요시 스스로 수정하세요. 마지막 결과물을 토대로 '페이지 N:' 형식의 최종 설명문을 작성해 주세요."""
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_content + "\n\n" + user_instruction,
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
