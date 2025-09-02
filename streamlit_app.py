
"""Streamlit interface for the slide explainer application."""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import streamlit as st

# Ensure the repository is on the Python path so package imports work
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from slide_explainer_app_advanced import (
    document_builder,
    llm_handler,
    pdf_processor,
)

st.set_page_config(page_title="Slide Explainer", layout="wide")
st.title("Slide Explainer")

# Inputs
api_key = st.text_input("OpenAI API Key", type="password")
uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
title = st.text_input("Document Title", value="Slide Explanations")
mode = st.selectbox("Mode", ["explain", "summarize"], index=0)
model = st.text_input("Model", value="gpt-5-mini-2025-08-07")
temperature_opt = st.text_input("Temperature (leave blank for default)", value="")
temperature = None
if temperature_opt.strip():
    try:
        temperature = float(temperature_opt)
    except ValueError:
        st.warning("Temperature must be a number. Using model default.")

section_size_limit = st.number_input("Section chunk size", 1, 20, 8, 1)

DEBUG_LOG = ROOT / "debug_output.txt"

# Initialize session state
st.session_state.setdefault("document", "")
st.session_state.setdefault("file_name", "")

generate = st.button("Generate")

def _write_debug(msg: str) -> None:
    with DEBUG_LOG.open("a", encoding="utf-8") as dbg:
        dbg.write(msg.rstrip() + "\n")

if generate:
    if not uploaded_pdf:
        st.error("Please upload a PDF file.")
    elif not api_key:
        st.error("Please provide an OpenAI API key.")
    else:
        # Clear previous debug
        if DEBUG_LOG.exists():
            DEBUG_LOG.unlink()

        os.environ["OPENAI_API_KEY"] = api_key

        try:
            with st.spinner("슬라이드를 분석하고 있습니다... 잠시만 기다려 주세요."):
                # Persist upload to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.getbuffer())
                    tmp_path = tmp.name

                # 1) Extract rich text + optional OCR
                page_metas: List[dict] = []
                with fitz.open(tmp_path) as doc:  # type: ignore[name-defined]
                    for i in range(len(doc)):
                        meta = pdf_processor.extract_page_text_with_layout(tmp_path, i)
                        meta = pdf_processor.conditional_ocr_page_image(
                            tmp_path,
                            i,
                            meta,
                            dpi=400,
                            lang="kor+eng",
                            ocr_threshold=300,
                        )
                        page_metas.append(meta)

                # 2) Segment into sections and detect duplicates
                sections, duplicates = pdf_processor.segment_into_sections(
                    tmp_path,
                    page_metas,
                    api_key,
                    semantic_weight=0.4,
                    visual_weight=0.2,
                    title_weight=0.4,
                    duplicate_phash_threshold=8,
                )
                _write_debug(f"Detected {len(sections)} sections; duplicates: {duplicates}")

                # 3) Generate outputs
                section_outputs: List[Tuple[str, List[Tuple[int, str]]]] = []

                if mode == "explain":
                    for section in sections:
                        pages = section.pages
                        # chunk pages to stay under token limits
                        chunks = [
                            pages[i : i + int(section_size_limit)]
                            for i in range(0, len(pages), int(section_size_limit))
                        ]
                        slides_accum: List[Tuple[int, str]] = []

                        for chunk in chunks:
                            items: List[Tuple[int, str]] = []
                            for p in chunk:
                                text = page_metas[p].get("body_text", "") or ""
                                if p in duplicates:
                                    canon = duplicates[p] + 1
                                    text = f"(중복) 이 슬라이드는 페이지 {canon}과 동일합니다."
                                items.append((p + 1, text))

                            _write_debug(f"Section: {section.title}")
                            _write_debug(f"Pages: {chunk}")

                            explanation = llm_handler.explain_section(
                                items,
                                section.title,
                                model=model,
                                language="ko",
                                max_completion_tokens=2200,
                                temperature=temperature,
                            )
                            _write_debug("Raw LLM output:")
                            _write_debug(explanation)

                            parsed = llm_handler.parse_page_explanations(explanation)
                            if parsed:
                                _write_debug(f"Matched {len(parsed)} page blocks")
                                slides_accum.extend(parsed)
                            else:
                                _write_debug("No '페이지 N:' matches; using paragraph fallback")
                                parts = [
                                    part.strip()
                                    for part in re.split(r"\n{2,}", explanation)
                                    if part.strip()
                                ]
                                for idx, p_num in enumerate(chunk):
                                    text = parts[idx] if idx < len(parts) else ""
                                    slides_accum.append((p_num + 1, text))

                        section_outputs.append((section.title, slides_accum))
                else:
                    # summarize mode
                    texts = [m.get("body_text", "") for m in page_metas]
                    groups = [list(range(len(texts)))]
                    summaries = llm_handler.summarize_groups(
                        api_key=api_key,
                        texts=texts,
                        groups=groups,
                        model=model,
                        temperature=temperature,
                    )
                    slides = [(i + 1, summaries.get(i, "")) for i in range(len(texts))]
                    section_outputs.append(("Summary", slides))

                # 4) Build final document
                document = document_builder.build_document(
                    title=title,
                    pdf_filename=os.path.basename(tmp_path),
                    sections=section_outputs,
                )
                st.session_state.document = document
                st.session_state.file_name = f"{title}.txt"
                st.success("설명 문서 생성이 완료되었습니다!")
                st.balloons()
        except Exception as exc:
            st.exception(exc)

# Preview / Download
if st.session_state.document:
    st.markdown("---")
    st.subheader("생성된 문서")
    preview_text = "\n".join(st.session_state.document.splitlines()[:20])
    st.text_area("미리보기", preview_text + "\n...", height=220)
    st.download_button(
        label="전체 내용 다운로드 (.txt)",
        data=st.session_state.document,
        file_name=st.session_state.file_name or "result.txt",
        mime="text/plain",
    )
