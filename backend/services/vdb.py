import os, json, math, datetime as dt
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient

COL = os.getenv("QDRANT_COLLECTION", "articles")

def _client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url)               # 원격 서버(도커 등)
    return QdrantClient(path="qdrant_local")       # 로컬 파일 기반 DB

def ingest_dummy(path: str) -> int:
    cli = _client()
    docs = json.loads(Path(path).read_text(encoding="utf-8"))

    texts, metas = [], []
    for d in docs:
        texts.append(d["body"])
        metas.append({
            "doc_id": d["id"], "title": d["title"], "url": d["url"],
            "published_at": d.get("published_at", "1970-01-01T00:00:00Z"),
            "source": d.get("source", "demo"),
            "text": d["body"],
            "tags": d.get("tags", [])
        })
    # FastEmbed로 로컬 임베딩 후 upsert까지 수행(서버/로컬 모두 동작)
    cli.add(collection_name=COL, documents=texts, metadata=metas)
    return len(texts)

def search_by_text(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    cli = _client()
    res = cli.query(collection_name=COL, query_text=query, limit=limit)
    out = []
    for p in res.points:
        payload = p.payload or {}
        payload["score"] = float(p.score)
        out.append(payload)
    return out

def _age_hours(iso_str: str) -> float:
    try:
        t = dt.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return max((dt.datetime.now(dt.timezone.utc) - t).total_seconds() / 3600.0, 0.0)
    except Exception:
        return 9999.0

def rank_for_reco(cands: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    # 간단 하이브리드: 의미 유사도(score) 70% + 신선도(half-life 36h) 30%
    ranked = []
    for it in cands:
        age_h = _age_hours(it.get("published_at", "1970-01-01T00:00:00Z"))
        fresh = math.exp(-age_h / 36.0)
        final = 0.7 * float(it.get("score", 0.0)) + 0.3 * fresh
        ranked.append({
            "title": it.get("title"), "url": it.get("url"),
            "published_at": it.get("published_at"), "score": final,
            "reason": "의미 유사도 + 신선도"
        })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:limit]