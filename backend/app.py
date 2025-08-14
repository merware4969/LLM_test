import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal

# 미적용(베이스라인)에서 사용하는 vdb 유틸
from services.vdb import ingest_dummy, search_by_text, rank_for_reco
from services.rag import make_briefing
from services.compare import run_compare

# ----- [선택] reco.py(하이브리드 휴리스틱) 적용을 위한 준비 -----
# reco.py를 적용하려면 아래 주석을 해제하고, RECO_FN_MAP의 "hybrid" 주석도 해제하세요.
# try:
#     from services.reco import get_recommendations as reco_hybrid
# except Exception:
#     reco_hybrid = None

app = FastAPI(title="LLM Sandbox (FastAPI)")

# 기본 추천 엔진: 미적용(simple). .env에서 RECO_ENGINE=simple|raw|hybrid 로 바꿀 수 있습니다.
RECO_ENGINE_DEFAULT = os.getenv("RECO_ENGINE", "simple").lower()

class IngestIn(BaseModel):
    path: str = "backend/data/dummy_articles.json"

class QueryIn(BaseModel):
    user_id: Optional[str] = None
    query: str
    top_k: int = 5   # 요약용 컨텍스트 수
    top_n: int = 10  # 추천 개수
    # 요청 단위 토글(없으면 환경변수 기본값 사용)
    reco_mode: Optional[Literal["simple", "raw", "hybrid"]] = None

class CompareIn(BaseModel):
    prompt: str
    candidates: List[Dict[str, str]]  # [{"provider":"gemini","model":"gemini-2.0-flash"}, ...]

@app.post("/ingest")
def ingest(p: IngestIn):
    n = ingest_dummy(p.path)
    return {"ok": True, "count": n}

# ----- 추천 엔진 구현들 -----
def reco_simple(user_id: Optional[str], query: str, top_n: int):
    """
    미적용(베이스라인): vdb에서 후보 검색(유사도) → 유사도+신선도(0.7/0.3) 가중 rank_for_reco 사용
    """
    candidates = search_by_text(query, limit=max(top_n * 2, 10))
    return rank_for_reco(candidates, limit=top_n)

def reco_raw(user_id: Optional[str], query: str, top_n: int):
    """
    완전 단순: 유사도(score)만으로 상위 N개
    """
    cands = search_by_text(query, limit=top_n)
    return [
        {
            "title": p.get("title"),
            "url": p.get("url"),
            "published_at": p.get("published_at"),
            "score": float(p.get("score", 0.0)),
            "reason": "의미 유사도 점수만 사용(raw)"
        }
        for p in cands
    ]

# reco.py 하이브리드 적용 (주석 해제 시 활성화)
# def reco_hybrid_wrapped(user_id: Optional[str], query: str, top_n: int):
#     """
#     reco.py 적용: 유사도+신선도+출처가중치+품질(+인기도) & 다양성 보장
#     """
#     if reco_hybrid is None:
#         # reco.py를 아직 넣지 않았거나 import 실패 시 simple로 폴백
#         return reco_simple(user_id, query, top_n)
#     return reco_hybrid(user_id, query, top_n=top_n)

RECO_FN_MAP = {
    "simple": reco_simple,      # 기본값(미적용)
    "raw":    reco_raw,
    # "hybrid": reco_hybrid_wrapped,   # reco.py 적용 시 주석 해제
}

@app.post("/query")
def query(p: QueryIn):
    # 요청 단위(reco_mode) → 없으면 환경변수(RECO_ENGINE) → 없으면 simple
    engine = (p.reco_mode or RECO_ENGINE_DEFAULT)
    reco_fn = RECO_FN_MAP.get(engine, RECO_FN_MAP["simple"])

    # 좌: 추천
    recommendations = reco_fn(p.user_id, p.query, p.top_n)

    # 우: RAG 요약
    briefing = make_briefing(query=p.query, k=p.top_k)

    return {
        "engine": engine,
        "recommendations": recommendations,
        "briefing": briefing
    }

@app.post("/compare")
def compare(p: CompareIn):
    return run_compare(prompt=p.prompt, candidates=p.candidates)

# 아예 FastAPI가 프론트를 정적 서빙(추천)
from fastapi.staticfiles import StaticFiles
app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")
# 이제 http://127.0.0.1:8000/ui 로 접속하면 index.html이 뜹니다.
