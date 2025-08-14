from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm import get_chat
from .vdb import search_by_text

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "너는 기술 뉴스 요약 비서다. 제공된 컨텍스트로만 사실을 말해라. 5문장 이내로, 마지막 줄은 한줄 인사이트."),
    ("human", "질의: {query}\n\n컨텍스트:\n{context}\n\n요약:")
])

def make_briefing(query: str, k: int = 5, provider: str | None = None) -> Dict:
    ctx = search_by_text(query, limit=k)
    context_text = "\n\n".join([f"[{i+1}] {c['title']}\n{c['text'][:1200]}" for i, c in enumerate(ctx)])
    llm = get_chat(provider=provider, temperature=0)
    answer = (PROMPT | llm | StrOutputParser()).invoke({"query": query, "context": context_text})
    sources = [{"title": c["title"], "url": c["url"], "snippet": c["text"][:240]} for c in ctx]
    bullets = [ln for ln in answer.splitlines() if ln][:5]
    return {"answer": answer, "bullets": bullets, "sources": sources}