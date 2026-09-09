"""Evidence-grounded answer generation."""

from __future__ import annotations

from typing import Any

from src.llm import Tier, call_llm


async def generate_answer(query: str, evidence: list[dict[str, Any]]) -> str:
    """Write a response grounded only in supplied evidence, with [S#] citations."""
    context = "\n\n".join(f"[S{index}] {item['text']}" for index, item in enumerate(evidence, start=1))
    prompt = f"""Answer the research question using only the evidence below. Cite every factual statement inline as [S#].
If evidence is insufficient, say so plainly. Do not invent sources.

Question: {query}

Evidence:
{context}"""
    try:
        return (await call_llm(Tier.REASONING, [{"role": "user", "content": prompt}], temperature=0))["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Insufficient verified evidence was available to answer this question."
