"""LLM planning helpers for bounded research workflows."""

from __future__ import annotations

import json

from src.llm import Tier, call_llm


def _content(response: dict) -> str:
    return response["choices"][0]["message"]["content"]


async def plan_query(query: str) -> list[str]:
    """Use one reasoning call to decompose a research query into 2-4 questions."""
    prompt = f"""Decompose this research request into 2 to 4 focused, answerable sub-questions.
Return ONLY a JSON array of strings; do not answer the questions.

Request: {query}"""
    try:
        items = json.loads(_content(await call_llm(Tier.REASONING, [{"role": "user", "content": prompt}], temperature=0)))
        if isinstance(items, list):
            questions = [item.strip() for item in items if isinstance(item, str) and item.strip()]
            if questions:
                return questions[:4]
    except Exception:
        pass
    return [query]


async def rewrite_question(question: str) -> str:
    """Make one bounded retrieval-oriented rewrite of a weak sub-question."""
    prompt = f"""Rewrite this as one more specific factual search question for scholarly retrieval.
Return only the rewritten question, with no commentary.

Question: {question}"""
    try:
        rewritten = _content(await call_llm(Tier.REASONING, [{"role": "user", "content": prompt}], temperature=0)).strip()
        return rewritten or question
    except Exception:
        return question
