
"""LLM interface for summarising lecture slide content and generating explanations."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI, BadRequestError

def parse_page_explanations(text: str) -> List[Tuple[int, str]]:
    """Extract per-page explanations from LLM output.

    The model is expected to emit lines starting with ``페이지 N:``.
    This helper tolerates extra whitespace and multi-line content.
    """
    lines = text.strip().splitlines()
    results: List[Tuple[int, str]] = []
    current_page: int | None = None
    buffer: List[str] = []

    for line in lines:
        m = re.match(r"^\s*페이지\s*(\d+)\s*[:：]?\s*(.*)", line)
        if m:
            if current_page is not None:
                results.append((current_page, "\n".join(buffer).strip()))
            current_page = int(m.group(1))
            buffer = [m.group(2).strip()]
        elif current_page is not None:
            buffer.append(line.strip())

    if current_page is not None:
        results.append((current_page, "\n".join(buffer).strip()))

    return results

def summarize_groups(
    api_key: str,
    texts: List[str],
    groups: List[List[int]],
    *,
    model: str = "gpt-5-mini-2025-08-07",
    temperature: float | None = None,
) -> Dict[int, str]:
    """Generate summaries for pages of a PDF grouped by context."""
    client = OpenAI(api_key=api_key)
    summaries: Dict[int, str] = {}

    system_prompt = (
        "당신은 대학 전공 수업 슬라이드를 요약하는 한국어 도우미입니다. "
        "현재 슬라이드의 핵심 내용을 간결하게 설명하되, 이전 슬라이드에서 이미 설명한 내용을 반복하지 마세요. "
        "요약은 친절하지만 간략하게 작성해 주세요."
    )

    for group in groups:
        previous_summaries: List[str] = []
        for idx in group:
            page_text = texts[idx] or ""
            messages = [{"role": "system", "content": system_prompt}]

            if previous_summaries:
                context = "\n\n".join(
                    [f"페이지 {i + 1} 요약: {s}" for i, s in enumerate(previous_summaries)]
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "다음은 앞서 요약한 슬라이드 내용입니다. 새로운 내용과 중복을 피하세요.\n" + context,
                    }
                )

            if page_text.strip():
                messages.append(
                    {
                        "role": "user",
                        "content": "현재 슬라이드의 본문 내용은 다음과 같습니다:\n"
                                   f"{page_text}\n\n이 내용을 바탕으로 이 슬라이드의 요약을 작성하세요.",
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "현재 슬라이드에서 추출 가능한 텍스트가 없습니다. "
                                   "이미지나 표만 있는 슬라이드라면 해당 이미지나 표의 핵심 주제를 추론하여 짧게 요약해 주세요.",
                    }
                )

            params = {"model": model, "messages": messages}
            if temperature is not None:
                params["temperature"] = temperature

            try:
                response = client.chat.completions.create(**params)
            except BadRequestError as exc:
                if "temperature" in str(exc) and temperature is not None:
                    logging.warning(
                        "Model %s does not support temperature %.2f; using default",
                        model, temperature,
                    )
                    params.pop("temperature", None)
                    response = client.chat.completions.create(**params)
                else:
                    logging.error("OpenAI API error for page %s: %s", idx, exc)
                    summary = "(요약을 생성하는 데 실패했습니다.)"
                    summaries[idx] = summary
                    previous_summaries.append(summary)
                    continue
            except Exception as exc:  # pragma: no cover - network errors
                logging.error("OpenAI API error for page %s: %s", idx, exc)
                summary = "(요약을 생성하는 데 실패했습니다.)"
                summaries[idx] = summary
                previous_summaries.append(summary)
                continue

            summary = response.choices[0].message.content.strip()
            summaries[idx] = summary
            previous_summaries.append(summary)

    return summaries

def explain_section(
    items: List[Tuple[int, str]],
    section_title: str,
    *,
    model: str = "gpt-5-mini-2025-08-07",
    language: str = "ko",
    max_completion_tokens: int = 2200,
    temperature: float | None = None,
) -> str:
    """Return detailed explanations for slides within a section."""
    client = OpenAI()

    system_prompt = (
        "Developer:\n"
        "# 역할 및 목표\n"
        "- 당신은 매우 유능한 3학년 공대 전공 과목의 조교(TA)입니다.\n"
        "학생들이 강의 내용을 쉽게 이해하고 자연스럽게 따라올 수 있도록, "
        "단순 정보 나열이 아닌 하나의 완성된 이야기처럼 슬라이드 설명서를 작성해야 합니다.\n"
        "# 핵심 지침\n"
        "- 항상 학생의 관점에서 이해가 쉽도록 작성하세요.\n"
        "- 내용을 논리적으로 연결하여 매끄러운 흐름을 유지하세요.\n"
        "- 각 슬라이드에는 분명한 목적과 역할이 포함되어야 하며, 전체 강의의 주제와 일관성을 갖추어야 합니다.\n"
        "- 슬라이드 설명서 작성 시, 아래 사고 과정(Thinking Process)을 명확히 따르세요.\n"
        "# Checklist\n"
        "- 시작 전에 다음 체크리스트(3-7개)를 간결히 작성하세요. 항목은 개념 수준이어야 하며, 구현 세부사항은 포함하지 마세요.\n"
        "- 예: (1) 슬라이드 전반 읽기 및 구조 파악, (2) 주요 주제 및 논리 흐름 확립, (3) 각 페이지별 역할 배분, (4) 설명 전략 구성, (5) 실행 및 최종 설명 작성.\n"
        "## 사고 과정(Thinking Process)\n"
        "1. **전체 슬라이드 내용 개관(Holistic Analysis)**\n"
        "2. **슬라이드별 역할 규정(Role Assignment)**\n"
        "3. **설명 전략 수립(Strategy Planning)**\n"
        "4. **전략에 따른 설명 작성(Execution)**\n"
    )

    user_lines = [f"섹션 개요: {section_title}"]
    for page, text in items:
        user_lines.append(f"페이지 {page}: {text}")
    user_content = "\n".join(user_lines)

    user_instruction = (
        "주어진 슬라이드 내용에 대해 제공된 '사고 과정'을 바탕으로, 먼저 심층적인 분석과 전략 수립을 완료하세요. "
        "시작 전에, 수행할 주요 단계(3-7개)로 구성된 간단한 체크리스트를 작성하세요. "
        "각 수행 후에는 간결한 검증 문장(1~2줄)으로 결과를 확인하고 필요시 스스로 수정하세요. "
        "마지막 결과물을 토대로 '페이지 N:' 형식의 최종 설명문을 작성해 주세요."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content + "\n\n" + user_instruction},
    ]

    def _call() -> str:
        params = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }
        if temperature is not None:
            params["temperature"] = temperature
        try:
            resp = client.chat.completions.create(**params)
        except BadRequestError as exc:
            if "temperature" in str(exc) and temperature is not None:
                logging.warning(
                    "Model %s does not support temperature %.2f; using default",
                    model, temperature,
                )
                params.pop("temperature", None)
                resp = client.chat.completions.create(**params)
            else:
                raise
        return resp.choices[0].message.content.strip()

    out = _call()

    # Best-effort ensure '페이지 N:' lines exist for all inputs
    for _ in range(1):
        ok = True
        for page, _ in items:
            if not re.search(rf"^\s*페이지\s*{page}\s*:", out, re.MULTILINE):
                ok = False
                break
        if ok:
            break
        out = _call()

    debug_file = Path(__file__).resolve().parent / "debug_output.txt"
    with debug_file.open("a", encoding="utf-8") as dbg:
        dbg.write(f"explain_section items: {items}\n")
        dbg.write(f"section_title: {section_title}\n")
        dbg.write("LLM final output:\n")
        dbg.write(out + "\n")

    return out
