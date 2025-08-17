"""Entry point for the advanced lecture slide summarisation application.

This script leverages enhanced segmentation and deduplication to
improve the quality of slide summaries. It detects topic boundaries
using semantic and visual changes and avoids summarising nearly
identical slides multiple times. The final summaries are written to
Google Docs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

from . import pdf_processor
from . import llm_handler
from . import google_docs_writer


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced summarise lecture slides and save to Google Docs"
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF file containing lecture slides",
    )
    parser.add_argument(
        "--openai-key",
        required=False,
        help="OpenAI API key. If omitted, uses the OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--google-creds",
        required=False,
        help="Path to Google service account key or OAuth client secrets JSON",
    )
    parser.add_argument(
        "--title",
        default="Lecture Summary (Advanced)",
        help="Title for the generated Google Document",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=2,
        help="Number of groups to divide the slides into (default: 2)",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="OpenAI model to use for summarisation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for the OpenAI model",
    )
    parser.add_argument(
        "--share-email",
        default=None,
        help="Email to share the resulting document with (service account only)",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model for semantic analysis",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=None,
        help="Approximate proportion of pages for the first group (0 < ratio < 1)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pdf_path = args.pdf
    if not os.path.isfile(pdf_path):
        logging.error("PDF file not found: %s", pdf_path)
        return 1
    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        logging.error("OpenAI API key not provided.")
        return 1

    # Segment slides using the advanced algorithm
    logging.info("Segmenting PDF into groups using advanced algorithm...")
    try:
        groups, texts, duplicates = pdf_processor.advanced_split_pdf(
            pdf_path,
            api_key=openai_key,
            expected_groups=args.groups,
            target_ratio=args.target_ratio,
            embedding_model=args.embedding_model,
        )
    except Exception as exc:
        logging.error("Failed to process PDF: %s", exc)
        return 1
    logging.info("Identified %d group(s) and %d duplicate page(s)", len(groups), len(duplicates))

    # Determine which pages to summarise (exclude duplicates)
    unique_page_indices = [i for i in range(len(texts)) if i not in duplicates]
    # Build new groups referencing only unique pages for summarisation order
    summarisation_groups = []
    for group in groups:
        summarisation_groups.append([i for i in group if i in unique_page_indices])

    # Summarise unique pages
    logging.info("Generating summaries using OpenAI (%s)...", args.model)
    summaries = llm_handler.summarize_groups(
        api_key=openai_key,
        texts=texts,
        groups=summarisation_groups,
        model=args.model,
        temperature=args.temperature,
    )

    # Populate summaries for duplicates by copying from their canonical page
    for dup, canonical in duplicates.items():
        if canonical in summaries:
            summaries[dup] = summaries[canonical] + " (중복 슬라이드)"
        else:
            summaries[dup] = summaries.get(dup, "(중복 슬라이드)")

    # Create Google Doc
    logging.info("Writing summaries to Google Docs...")
    try:
        doc_id = google_docs_writer.create_document_from_summaries(
            credentials_path=args.google_creds,
            title=args.title,
            summaries=summaries,
            share_email=args.share_email,
        )
    except Exception as exc:
        logging.error("Failed to create Google Document: %s", exc)
        return 1

    logging.info(
        "Document created: https://docs.google.com/document/d/%s/edit", doc_id
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())