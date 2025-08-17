# slide_explainer_app_advanced/auto_split.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import imagehash
from pdf2image import convert_from_path

from .embedding_utils import embed_texts  # 이미 프로젝트에 포함됨

@dataclass
class AutoSegConfig:
    min_seg_len: int = 3         # 최소 구간 길이(페이지)
    max_groups: int = 6          # 자동 분할 시 최대 그룹 수
    smooth_window: int = 3       # 경계 점수 평활화 창 크기(홀수 권장)
    cut_penalty: Optional[float] = None  # 컷 1개당 벌점(없으면 자동 산정)

def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1: return x
    k = w // 2
    pad = np.pad(x, (k, k), mode="edge")
    ker = np.ones(w) / w
    return np.convolve(pad, ker, mode="valid")

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _phash_distance(img1: Image.Image, img2: Image.Image) -> float:
    # 0~1 정규화된 pHash 해밍거리
    h1 = imagehash.phash(img1); h2 = imagehash.phash(img2)
    return (h1 - h2) / (len(h1.hash) * len(h1.hash))

def compute_boundary_scores(
    pdf_path: str, page_texts: List[str], dpi: int = 110, embedding_model: str = "text-embedding-3-small"
) -> np.ndarray:
    """
    인접 페이지 경계(1|2, 2|3, ..., N-1|N)의 점수.
    의미 변화(1 - 코사인유사도) + 시각 변화(pHash) + 제목 변화(불일치=1) 가중합.
    """
    N = len(page_texts)
    if N < 2: return np.array([])
    # 의미 변화
    vecs = embed_texts(page_texts, model=embedding_model)
    sem = np.array([1.0 - _cosine_sim(vecs[i-1], vecs[i]) for i in range(1, N)])

    # 시각 변화
    imgs = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=N)
    vis = np.array([_phash_distance(imgs[i-1], imgs[i]) for i in range(1, N)])

    # 제목 변화(간단 규칙: 첫 줄 비교)
    def title_of(t: str) -> str:
        if not t: return ""
        line = t.strip().splitlines()[0] if t.strip().splitlines() else ""
        return line.lower().strip()[:40]
    titles = [title_of(t) for t in page_texts]
    ttl = np.array([0.0 if titles[i-1] == titles[i] else 1.0 for i in range(1, N)])

    # 가중치(기본: 의미 0.5, 시각 0.3, 제목 0.2)
    w1, w2, w3 = 0.5, 0.3, 0.2
    B = w1 * sem + w2 * vis + w3 * ttl
    return B

def _ok_to_add(pos: int, cand: List[int], N: int, min_seg_len: int) -> bool:
    all_cuts = sorted(cand + [pos])
    prev = 1
    for c in all_cuts:
        seg_len = c - prev + 1
        if seg_len < min_seg_len: return False
        prev = c + 1
    if (N - prev + 1) < min_seg_len: return False
    return True

def auto_choose_boundaries(B: np.ndarray, cfg: AutoSegConfig, N: int) -> List[int]:
    """
    자동 컷 선택:
      - 경계 점수 평활화
      - 컷 1개당 벌점 적용
      - 최소 구간 길이/최대 그룹 수 제약
    """
    if N < 2: return []
    S = _smooth(B, cfg.smooth_window)
    penalty = cfg.cut_penalty if cfg.cut_penalty is not None else float(np.median(S) * 0.8)

    candidates = sorted(range(1, N), key=lambda i: S[i-1], reverse=True)
    cuts: List[int] = []
    for c in candidates:
        if len(cuts) >= cfg.max_groups - 1: break
        gain = S[c-1] - penalty
        if gain <= 0:  # 이득 없으면 넘김
            continue
        if _ok_to_add(c, cuts, N, cfg.min_seg_len):
            cuts.append(c)
    cuts.sort()
    return cuts

def choose_fixed_boundaries(B: np.ndarray, N: int, groups: int, min_seg_len: int = 2) -> List[int]:
    """
    고정 그룹 수(G)가 주어진 경우, 상위 컷을 제약 아래에서 골라 G-1개 선택.
    """
    groups = max(2, groups)
    need = groups - 1
    if N < groups: return []

    S = _smooth(B, 3)
    candidates = sorted(range(1, N), key=lambda i: S[i-1], reverse=True)
    cuts: List[int] = []
    for c in candidates:
        if len(cuts) >= need: break
        if _ok_to_add(c, cuts, N, min_seg_len):
            cuts.append(c)
    cuts.sort()
    # 못 채웠으면 최소 길이 완화 재시도
    relax = min_seg_len
    while len(cuts) < need and relax > 1:
        relax -= 1
        cuts = []
        for c in candidates:
            if len(cuts) >= need: break
            if _ok_to_add(c, cuts, N, relax):
                cuts.append(c)
        cuts.sort()
    return cuts

def segments_from_cuts(N: int, cuts: List[int]) -> List[Tuple[int, int]]:
    """컷 리스트를 (start,end) 1-indexed 구간 리스트로 변환"""
    if not cuts: return [(1, N)]
    segs = []
    prev = 1
    for c in cuts:
        segs.append((prev, c))
        prev = c + 1
    segs.append((prev, N))
    return segs
