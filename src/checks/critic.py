"""LLM-backed grading of whether a source actually supports a claim."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ValidationError

from src.llm import Tier, call_llm


class EvidenceGrade(BaseModel):
    verdict: Literal["supports", "contradicts", "unclear"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str


def _prompt(claim: str, source_text: str, strict: bool = False) -> str:
    suffix = " Return ONLY valid JSON, with no markdown, explanation, or surrounding text." if strict else " Return valid JSON only."
    return f"""Act as a strict evidence grader. Decide whether the source text supports, contradicts, or leaves unclear the claim.
Confident-sounding claims without direct textual support must be graded 'unclear'; prefer 'unclear' over guessing or filling gaps from outside knowledge.
Use 'contradicts' only when the source clearly states an incompatible fact.
Return exactly this JSON schema: {{"verdict":"supports|contradicts|unclear","confidence":"high|medium|low","reasoning":"one sentence"}}.{suffix}

Claim: {claim}

Source text: {source_text}"""


def _parse(response: dict) -> EvidenceGrade:
    try:
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("model content is not text")
        return EvidenceGrade.model_validate(json.loads(content))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"invalid evidence-grader JSON: {error}") from error


async def grade_evidence(claim: str, source_text: str) -> EvidenceGrade:
    """Grade a claim against source text using the FAST tier.

    A malformed or schema-invalid first response gets one JSON-only retry.
    """
    for strict in (False, True):
        response = await call_llm(
            Tier.FAST,
            [{"role": "user", "content": _prompt(claim, source_text, strict)}],
            temperature=0,
        )
        try:
            return _parse(response)
        except ValueError:
            if strict:
                raise
    raise AssertionError("unreachable")
