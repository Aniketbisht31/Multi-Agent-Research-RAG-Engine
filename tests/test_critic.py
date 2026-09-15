import json

import pytest

from src.checks import critic


@pytest.fixture
def anyio_backend():
    return "asyncio"


def mock_response(verdict: str, confidence: str = "high") -> dict:
    return {
        "choices": [
            {"message": {"content": json.dumps({"verdict": verdict, "confidence": confidence, "reasoning": "The source directly establishes this."})}}
        ]
    }


@pytest.mark.anyio
async def test_grade_evidence_supports_clear_claim(monkeypatch):
    async def fake_call(*args, **kwargs):
        return mock_response("supports")

    monkeypatch.setattr(critic, "call_llm", fake_call)
    grade = await critic.grade_evidence("Water freezes at 0 C.", "At standard pressure, water freezes at 0 C.")
    assert grade.verdict == "supports"


@pytest.mark.anyio
async def test_grade_evidence_marks_different_topic_unclear(monkeypatch):
    async def fake_call(*args, **kwargs):
        return mock_response("unclear", "low")

    monkeypatch.setattr(critic, "call_llm", fake_call)
    grade = await critic.grade_evidence("Mars has liquid oceans.", "Honeybees communicate locations through waggle dances.")
    assert grade.verdict == "unclear"


@pytest.mark.anyio
async def test_grade_evidence_detects_contradiction(monkeypatch):
    async def fake_call(*args, **kwargs):
        return mock_response("contradicts")

    monkeypatch.setattr(critic, "call_llm", fake_call)
    grade = await critic.grade_evidence("The study involved 500 participants.", "The study enrolled 50 participants.")
    assert grade.verdict == "contradicts"
