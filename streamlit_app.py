"""Streamlit interface for the slide explainer application."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository is on the Python path so package imports work
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

import os
import re
import tempfile
from typing import List, Tuple

import fitz  # type: ignore
import streamlit as st

from slide_explainer_app_advanced import (
    document_builder,
    llm_handler,
    pdf_processor,
)



st.title("Slide Explainer")

api_key = st.text_input("OpenAI API Key", type="password")
uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
title = st.text_input("Document Title", value="Slide Explanations")
mode = st.selectbox("Mode", ["explain", "summarize"], index=0)
DEBUG_LOG = ROOT / "debug_output.txt"

# Initialize session state for storing results
if "document" not in st.session_state:
    st.session_state.document = None
if "file_name" not in st.session_state:
    st.session_state.file_name = ""

generate = st.button("Generate")
if generate and uploaded_pdf and api_key:
    if DEBUG_LOG.exists():
        DEBUG_LOG.unlink()
    os.environ["OPENAI_API_KEY"] = api_key
    with st.spinner("슬라이드를 분석하고 있습니다... 잠시만 기다려 주세요."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.getbuffer())
            tmp_path = tmp.name

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
                with DEBUG_LOG.open("a", encoding="utf-8") as dbg:
                    dbg.write(f"Section: {section.title}\n")
                    dbg.write(f"Pages: {chunk}\n")
                    dbg.write("Raw LLM output:\n")
                    dbg.write(explanation + "\n")

                # 기존에는 "페이지 N:" 형식이 조금이라도 어긋나면 슬라이드가 모두 무시되었다.
                # 아래 로직은 해당 패턴을 느슨하게 매칭하고, 전혀 매칭되지 않으면
                # 문단 단위로 분할하여 페이지 순서대로 할당한다.
                pattern = re.compile(
                    r"페이지\s*(\d+)\s*[:：]?\s*(.*?)(?=\n\s*페이지\s*\d+\s*[:：]?|\Z)",
                    re.S,
                )
                matches = list(pattern.finditer(explanation))

                if matches:
                    with DEBUG_LOG.open("a", encoding="utf-8") as dbg:
                        dbg.write(f"Matched {len(matches)} page blocks\n")
                    for match in matches:
                        num = int(match.group(1))
                        txt = match.group(2).strip()
                        slides_accum.append((num, txt))
                else:
                    with DEBUG_LOG.open("a", encoding="utf-8") as dbg:
                        dbg.write("No '페이지 N:' matches; using paragraph fallback\n")
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
        texts = [m.get("body_text", "") for m in page_metas]
        groups = [list(range(len(texts)))]
        summaries = llm_handler.summarize_groups(
            api_key=api_key,
            texts=texts,
            groups=groups,
            model="gpt-5-mini",
            # Use the model's default temperature

        )
        st.session_state.document = document
        st.session_state.file_name = f"{title}.txt"

    st.success("설명 문서 생성이 완료되었습니다!")
    st.balloons()
elif generate and not uploaded_pdf:
    st.error("Please upload a PDF file.")
elif generate and not api_key:
    st.error("Please provide an OpenAI API key.")

# Show preview and download option when document exists
if st.session_state.document:
    st.markdown("---")
    st.subheader("🎉 생성된 문서")
    preview_text = "\n".join(st.session_state.document.splitlines()[:15])
    st.text_area("미리보기", preview_text + "\n...", height=200)
    st.download_button(
        label="📄 전체 내용 다운로드 (.txt)",
        data=st.session_state.document,
        file_name=st.session_state.file_name,
        mime="text/plain",
    )
