import time
from typing import List, Dict, Any
from .llm import get_chat

def run_compare(prompt: str, candidates: List[Dict[str, str]]) -> Dict[str, Any]:
    results = []
    for c in candidates:
        llm = get_chat(provider=c.get("provider"), model=c.get("model"), temperature=0)
        t0 = time.perf_counter()
        out = llm.invoke(prompt)
        t1 = time.perf_counter()
        meta = getattr(out, "response_metadata", {}) if hasattr(out, "response_metadata") else {}
        results.append({
            "provider": c.get("provider"), "model": c.get("model"),
            "latency_ms": int((t1 - t0) * 1000),
            "tokens_in": meta.get("input_tokens"), "tokens_out": meta.get("output_tokens"),
            "answer": str(out)[:800]
        })
    return {"prompt": prompt, "models": results}