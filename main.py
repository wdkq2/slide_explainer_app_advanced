"""Command line entry point for the slide explainer application."""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Tuple

import fitz  # type: ignore
from . import pdf_processor, llm_handler, document_builder


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detailed slide explanations")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--title", default="Slide Explanations", help="Document title")
    parser.add_argument("--openai-key", required=False, help="OpenAI API key")
    parser.add_argument("--output-dir", default=".", help="Directory to save the result")

    parser.add_argument("--mode", choices=["explain", "summarize"], default="explain")
    parser.add_argument("--lang", default="ko")
    parser.add_argument("--section-size-limit", type=int, default=8)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=None)

    parser.add_argument(
        "--max-completion-tokens",
        "--max-tokens",
        dest="max_completion_tokens",
        type=int,
        default=2200,
    )

    parser.add_argument("--semantic-weight", type=float, default=0.4)
    parser.add_argument("--visual-weight", type=float, default=0.2)
    parser.add_argument("--title-weight", type=float, default=0.4)
    parser.add_argument("--duplicate-threshold-phash", type=int, default=8)

    parser.add_argument("--ocr-threshold", type=int, default=300)
    parser.add_argument("--ocr-dpi", type=int, default=400)
    parser.add_argument("--lang-ocr", default="kor+eng")

    return parser.parse_args(argv)


def _load_openai_key(args: argparse.Namespace) -> str:
    key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OpenAI API key required")
    os.environ["OPENAI_API_KEY"] = key
    return key


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = _load_openai_key(args)

    logging.basicConfig(level=logging.INFO)

    # 1. Extract text with layout and optional OCR
    page_metas: List[dict] = []
    with fitz.open(args.pdf) as doc:  # type: ignore[name-defined]
        for i in range(len(doc)):
            meta = pdf_processor.extract_page_text_with_layout(args.pdf, i)
            meta = pdf_processor.conditional_ocr_page_image(
                args.pdf,
                i,
                meta,
                dpi=args.ocr_dpi,
                lang=args.lang_ocr,
                ocr_threshold=args.ocr_threshold,
            )
            page_metas.append(meta)

    # 2. Hybrid segmentation & duplicate detection
    sections, duplicates = pdf_processor.segment_into_sections(
        args.pdf,
        page_metas,
        api_key,
        semantic_weight=args.semantic_weight,
        visual_weight=args.visual_weight,
        title_weight=args.title_weight,
        duplicate_phash_threshold=args.duplicate_threshold_phash,
    )

    logging.info("Detected %d sections", len(sections))
    logging.info("Duplicate mapping: %s", duplicates)

    # 3. Generate explanations or summaries
    section_outputs: List[Tuple[str, List[Tuple[int, str]]]] = []
    if args.mode == "explain":
        for section in sections:
            pages = section.pages
            chunked = [
                pages[i : i + args.section_size_limit]
                for i in range(0, len(pages), args.section_size_limit)
            ]
            slides_accum: List[Tuple[int, str]] = []
            for chunk in chunked:
                items: List[Tuple[int, str]] = []
                for p in chunk:
                    text = page_metas[p].get("body_text", "")
                    if p in duplicates:
                        canon = duplicates[p] + 1
                        text = f"(중복) 이 슬라이드는 페이지 {canon}과 동일합니다."
                    items.append((p + 1, text))
                explanation = llm_handler.explain_section(
                    items,
                    section.title,
                    model=args.model,
                    language=args.lang,
                    max_completion_tokens=args.max_completion_tokens,
                    temperature=args.temperature,
                )
                import re

                pattern = re.compile(
                    r"페이지\s*(\d+)\s*:\s*(.*?)(?=\n\s*페이지\s*\d+\s*:|\Z)",
                    re.S,
                )
                for match in pattern.finditer(explanation.strip() + "\n"):
                    num = int(match.group(1))
                    txt = match.group(2).strip()
                    slides_accum.append((num, txt))
            section_outputs.append((section.title, slides_accum))
    else:
        texts = [m.get("body_text", "") for m in page_metas]
        groups = [list(range(len(texts)))]
        summaries = llm_handler.summarize_groups(
            api_key=api_key,
            texts=texts,
            groups=groups,
            model=args.model,
            temperature=args.temperature,
        )
        slides = [(i + 1, summaries[i]) for i in sorted(summaries.keys())]
        section_outputs.append(("Summary", slides))

    # 4. Save to local directory
    document = document_builder.build_document(
        title=args.title,
        pdf_filename=os.path.basename(args.pdf),
        sections=section_outputs,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.title}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(document)
    logging.info("Saved result to %s", output_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

