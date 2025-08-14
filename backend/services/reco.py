# services/reco.py
"""
개인화 베이스라인/비개인화 하이브리드 추천용 랭커
- 입력: query (문자열), user_id(옵션), top_n
- 내부: vdb.search_by_text()로 후보 수집 → 하이브리드 스코어링 → 다양성 보장 → Top-N
- 출력: [{title, url, published_at, score, reason}]
"""
from __future__ import annotations
import math
import datetime as dt
from typing import List, Dict, Any, Optional
from .vdb import search_by_text

# 출처(매체) 권위 가중치 예시(없으면 1.0)
SOURCE_AUTHORITY: Dict[str, float] = {
    "Example News": 1.0,
    "Example Tech": 0.95,
    "AI Digest": 0.9,
    "Dev Weekly": 0.9,
    "Search Tech": 0.85,
}

# 하이브리드 가중치 기본값
DEFAULT_WEIGHTS = {
    "sim": 0.60,    # 쿼리-문서 유사도
    "fresh": 0.20,  # 신선도(half-life 기반)
    "auth": 0.10,   # 출처 권위
    "qual": 0.05,   # 본문 품질(길이 등)
    "pop": 0.05,    # 인기도(없으면 0)
}

def _age_hours(iso_str: str) -> float:
    try:
        t = dt.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return max((dt.datetime.now(dt.timezone.utc) - t).total_seconds() / 3600.0, 0.0)
    except Exception:
        return 9999.0

def _freshness(age_h: float, half_life_h: float = 36.0) -> float:
    # half-life 감쇠; 0~1 범위
    return math.exp(-age_h / half_life_h)

def _authority(source: Optional[str]) -> float:
    if not source:
        return 0.8  # 정보 없음은 보수적으로
    return max(0.0, min(1.0, SOURCE_AUTHORITY.get(source, 0.9)))

def _quality(text: str) -> float:
    # 본문 길이 기반의 매우 단순한 품질 점수(0~1)
    n = len(text or "")
    # 600~1800자가 적당하다고 가정하여 0~1로 스케일링
    lo, hi = 300, 2000
    if n <= lo:
        return max(0.0, min(1.0, n / lo * 0.5))
    if n >= hi:
        return 1.0
    # 구간 선형 증가
    return 0.5 + 0.5 * ((n - lo) / (hi - lo))

def _popularity(payload: Dict[str, Any]) -> float:
    # 더미 데이터엔 없으므로 0 반환(실서비스 땐 클릭/공유/외부언급 등으로 채움)
    return float(payload.get("pop", 0.0))

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _fingerprint_title(title: Optional[str]) -> str:
    if not title:
        return ""
    t = "".join(ch.lower() for ch in title if ch.isalnum() or ch.isspace())
    # 너무 공격적으로 중복 제거되지 않도록 앞부분만 사용
    return t[:48].strip()

def _diversify(items: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """
    아주 단순한 다양성 보장:
    - 동일 제목 지문(fingerprint) 중복 제거
    - 동일 출처(source) 과다 노출 방지(최대 40% 선)
    """
    if not items:
        return []
    out: List[Dict[str, Any]] = []
    seen_titles = set()
    per_source = {}
    max_per_source = max(1, int(top_n * 0.4))

    # 1차 패스: 중복/출처 상한 고려
    for it in items:
        if len(out) >= top_n:
            break
        fp = _fingerprint_title(it.get("title"))
        if fp in seen_titles:
            continue
        src = (it.get("source") or "").strip()
        if per_source.get(src, 0) >= max_per_source:
            continue
        out.append(it)
        seen_titles.add(fp)
        per_source[src] = per_source.get(src, 0) + 1

    # 2차 보충: 아직 모자라면 남은 것에서 채움
    if len(out) < top_n:
        for it in items:
            if len(out) >= top_n:
                break
            fp = _fingerprint_title(it.get("title"))
            if fp in seen_titles:
                continue
            out.append(it)
            seen_titles.add(fp)
    return out

def _score_item(payload: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    # 유사도는 vdb.search_by_text의 score를 그대로 사용(0~1로 가정)
    sim = _clamp01(payload.get("score", 0.0))
    fresh = _freshness(_age_hours(payload.get("published_at", "1970-01-01T00:00:00Z")))
    auth = _authority(payload.get("source"))
    qual = _quality(payload.get("text", ""))
    pop  = _clamp01(_popularity(payload))

    w = weights
    final = (w["sim"] * sim + w["fresh"] * fresh + w["auth"] * auth +
             w["qual"] * qual + w["pop"] * pop)

    # UI에 보여줄 근거 문구
    reason = f"유사도 {sim:.2f}, 신선도 {fresh:.2f}, 출처 {auth:.2f}, 품질 {qual:.2f}"
    if pop > 0:
        reason += f", 인기도 {pop:.2f}"

    return {
        "title": payload.get("title"),
        "url": payload.get("url"),
        "published_at": payload.get("published_at"),
        "source": payload.get("source"),
        "score": float(final),
        "reason": reason,
    }

def get_recommendations(
    user_id: Optional[str],
    query: str,
    top_n: int = 10,
    *,
    candidate_factor: int = 3,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> List[Dict[str, Any]]:
    """
    추천 진입 함수.
    1) 쿼리 텍스트로 벡터 검색 → 후보 수집
    2) 하이브리드 점수 계산(유사도/신선도/출처/품질/인기도)
    3) 점수 정렬 후 간단 다양성 보장
    """
    # 후보는 top_n * candidate_factor 만큼 넉넉히
    k = max(top_n * candidate_factor, top_n + 5)
    candidates = search_by_text(query, limit=k)  # payload 목록

    scored = [_score_item(p, weights) for p in candidates]
    scored.sort(key=lambda x: x["score"], reverse=True)

    diversified = _diversify(scored, top_n=top_n)
    return diversified