# slide_explainer_app_advanced/main.py
from __future__ import annotations
import argparse
import os
from typing import Optional, List

from . import pdf_processor, llm_handler, drive_writer
from .auto_split import (
    AutoSegConfig, compute_boundary_scores, auto_choose_boundaries,
    choose_fixed_boundaries, segments_from_cuts
)

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced summarise lecture slides and save to Google Drive"
    )
    # 필수
    parser.add_argument("--pdf", required=True, help="Path to the PDF file containing lecture slides")
    parser.add_argument("--title", default="Lecture Summary (Advanced)", help="Title for the saved document")

    # 키/인증
    parser.add_argument("--openai-key", required=False,
                        help="OpenAI API key. If omitted, uses the OPENAI_API_KEY environment variable.")
    parser.add_argument("--drive-dir", default="/content/drive/MyDrive",
                        help="Directory in Google Drive to save the summary file")

    # 자동/수동 분할 옵션
    parser.add_argument("--groups", default="auto",
                        help="Number of groups or 'auto' (e.g., 'auto', '2', '3'). Default: 'auto'")
    parser.add_argument("--target-ratio", type=float, default=None,
                        help="Only used when groups=2. Approximate proportion of pages for the first group (0 < r < 1)")
    parser.add_argument("--min-seg-len", type=int, default=3,
                        help="Minimum pages per segment for auto mode (default: 3)")
    parser.add_argument("--max-groups", type=int, default=6,
                        help="Maximum number of groups for auto mode (default: 6)")
    parser.add_argument("--cut-penalty", type=float, default=None,
                        help="Penalty per cut for auto mode (higher=fewer cuts). If omitted, set automatically.")
    parser.add_argument("--embedding-model", default="text-embedding-3-small",
                        help="OpenAI embedding model for semantic analysis")

    # LLM 설정
    parser.add_argument("--model", default="gpt-3.5-turbo",
                        help="OpenAI model to use for summarisation")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature for the OpenAI model")
    return parser.parse_args(argv)

def _get_openai_key(args: argparse.Namespace) -> str:
    key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OpenAI API 키가 필요합니다. --openai-key 또는 OPENAI_API_KEY 환경변수를 설정하세요.")
    os.environ["OPENAI_API_KEY"] = key
    return key

def _get_texts(pdf_path: str) -> List[str]:
    # pdf_processor의 함수명이 다를 수도 있으므로 안전하게 처리
    if hasattr(pdf_processor, "extract_texts"):
        return pdf_processor.extract_texts(pdf_path)
    if hasattr(pdf_processor, "extract_text_from_pdf"):
        return pdf_processor.extract_text_from_pdf(pdf_path)
    raise SystemExit("pdf_processor에 텍스트 추출 함수가 없습니다. extract_texts 또는 extract_text_from_pdf를 확인하세요.")

def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    openai_key = _get_openai_key(args)

    # 1) PDF → 페이지 텍스트
    texts = _get_texts(args.pdf)
    N = len(texts)
    if N == 0:
        raise SystemExit("PDF에서 텍스트를 추출하지 못했습니다.")
    if N == 1:
        groups = [[0]]
    else:
        # 2) 자동/수동 분할
        groups_arg = args.groups.strip().lower() if isinstance(args.groups, str) else str(args.groups)
        if groups_arg == "auto":
            B = compute_boundary_scores(args.pdf, texts, embedding_model=args.embedding_model)
            cfg = AutoSegConfig(
                min_seg_len=max(2, args.min_seg_len),
                max_groups=max(2, args.max_groups),
                cut_penalty=args.cut_penalty,
                smooth_window=3,
            )
            cuts = auto_choose_boundaries(B, cfg, N)
            segs = segments_from_cuts(N, cuts)      # 1-indexed 구간
            groups = [list(range(s-1, e)) for (s, e) in segs]  # 0-indexed 페이지 인덱스
            print(f"[auto] 선택된 그룹 수 = {len(groups)}, 경계(페이지 인덱스 기준) = {cuts}")
        else:
            try:
                G = int(groups_arg)
            except ValueError:
                raise SystemExit("--groups 는 'auto' 또는 정수여야 합니다.")
            G = max(2, G)

            # 고정 G분할: 경계 점수로 상위 컷 G-1개 선택
            B = compute_boundary_scores(args.pdf, texts, embedding_model=args.embedding_model)

            if G == 2 and args.target_ratio:
                # 2분할 + 비율 유도: 목표 위치 부근 컷 우대
                import numpy as np
                target = max(1, min(N-1, int(round(N * float(args.target_ratio)))))
                S = np.array(B)  # 길이 N-1
                # 목표 위치와의 거리 패널티(거리가 멀수록 감점)
                dist = np.array([abs((i) - target) for i in range(1, N)])
                dist_pen = (dist / max(1, N/2)) * (np.median(S) * 0.8)
                score = S - dist_pen
                # 최소 길이 제약을 만족하는 최적 컷 선택
                best_k, best_val = None, -1e9
                for i in range(1, N):
                    left_len = i
                    right_len = N - i
                    if left_len >= max(2, args.min_seg_len) and right_len >= max(2, args.min_seg_len):
                        val = score[i-1]
                        if val > best_val:
                            best_k, best_val = i, val
                if best_k is None:
                    # 제약 만족 컷이 없으면 일반 컷 선택으로 폴백
                    cuts = choose_fixed_boundaries(B, N, groups=2, min_seg_len=max(2, args.min_seg_len))
                else:
                    cuts = [best_k]
            else:
                cuts = choose_fixed_boundaries(B, N, groups=G, min_seg_len=max(2, args.min_seg_len))

            segs = segments_from_cuts(N, cuts)
            groups = [list(range(s-1, e)) for (s, e) in segs]
            print(f"[fixed] 그룹 수 = {len(groups)}, 경계 = {cuts}")

    # 3) 요약 생성 (그룹 컨텍스트로 중복 최소화)
    summaries = llm_handler.summarize_groups(
        api_key=openai_key,
        texts=texts,
        groups=groups,
        model=args.model,
        temperature=args.temperature,
    )

    # 4) Google Drive에 저장
    file_path = drive_writer.save_document_to_drive(
        title=args.title,
        summaries=summaries,
        drive_dir=args.drive_dir,
    )
    print("Saved:", file_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
