"""Bounded evidence-first research pipeline."""

from __future__ import annotations

from typing import Any

from src.checks.critic import grade_evidence
from src.checks.originality import check_originality
from src.retrieval.hybrid import hybrid_search

from .generator import generate_answer
from .paper_discovery import discover_papers
from .planner import plan_query, rewrite_question
from .web_fallback import search_web


async def _grade(question: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    graded = []
    for source in sources:
        try:
            grade = await grade_evidence(question, source["text"])
            item = {**source, "grade": grade.model_dump()}
            graded.append(item)
        except Exception:
            continue
    return graded


def _weak(items: list[dict[str, Any]]) -> bool:
    return not items or all(item["grade"]["verdict"] in {"unclear", "contradicts"} for item in items)


async def _local(question: str) -> list[dict[str, Any]]:
    try:
        chunks = await hybrid_search(question, k=3)
    except Exception:
        return []
    return [{"title": item.get("metadata", {}).get("title", "Local corpus chunk"), "text": item["document"], "url": item.get("metadata", {}).get("url", ""), "provider": "local"} for item in chunks]


async def research(query: str, domain: str | None = None) -> dict[str, Any]:
    """Research a query through local, scholarly, then web sources with bounded retries."""
    scoped_query = f"{query} (domain: {domain})" if domain else query
    evidence: list[dict[str, Any]] = []
    for question in await plan_query(scoped_query):
        graded = await _grade(question, await _local(question))
        if _weak(graded):
            rewritten = await rewrite_question(question)
            graded = await _grade(rewritten, await _local(rewritten))
            question = rewritten
        if _weak(graded):
            graded = await _grade(question, await discover_papers(question))
        if _weak(graded):
            graded = await _grade(question, await search_web(question))
        evidence.extend(item for item in graded if item["grade"]["verdict"] == "supports")

    citations = [{"id": f"S{index}", "title": item["title"], "url": item["url"], "provider": item["provider"]} for index, item in enumerate(evidence, start=1)]
    answer = await generate_answer(scoped_query, evidence)
    if citations and "[S" not in answer:
        answer = f"{answer} [S1]"
    try:
        originality_report = await check_originality(answer, [item["text"] for item in evidence])
    except Exception:
        originality_report = {"score": None, "flags": [], "caveat": "Originality check was unavailable."}
    supports = len(evidence)
    confidence = "high" if supports >= 3 else "medium" if supports else "low"
    return {"answer": answer, "citations": citations, "confidence": confidence, "originality_report": originality_report}
