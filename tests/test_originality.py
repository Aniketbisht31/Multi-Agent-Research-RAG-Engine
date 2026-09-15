import pytest

from src.checks import originality


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_originality_flags_verbatim_and_paraphrase(monkeypatch):
    draft = (
        "The rapid brown fox leaps across the quiet meadow at sunrise. "
        "Engineers designed a compact solar system for lunar habitats."
    )
    sources = [
        "The rapid brown fox leaps across the quiet meadow at sunrise.",
        "Scientists built a small photovoltaic array to power homes on the moon.",
    ]

    async def fake_embed(_, texts):
        return [
            [1.0, 0.0]
            if "fox" in text.lower()
            else [0.0, 1.0]
            if ("solar" in text.lower() or "photovoltaic" in text.lower())
            else [0.0, 0.0]
            for text in texts
        ]

    monkeypatch.setattr(originality, "embed", fake_embed)
    result = await originality.check_originality(draft, sources)

    assert {flag["type"] for flag in result["flags"]} == {"verbatim", "paraphrase"}
    assert next(flag for flag in result["flags"] if flag["type"] == "verbatim")["matched_source"] == 0
    assert next(flag for flag in result["flags"] if flag["type"] == "paraphrase")["matched_source"] == 1
