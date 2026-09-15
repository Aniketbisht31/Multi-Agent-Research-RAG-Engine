import json

import httpx
import pytest

from src.checks import provenance


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("body", "http_status", "expected"),
    [
        ({"message": {"relation": {"is-retracted-by": [{"id": "x"}]} }}, 200, "retracted"),
        ({"message": {"relation": {"is-corrected-by": [{"id": "x"}]} }}, 200, "corrected"),
        ({"message": {"title": ["An ordinary paper"]}}, 200, "clean"),
        ({"status": "resource-not-found"}, 404, "unverified"),
    ],
)
async def test_crossref_status_branches(monkeypatch, tmp_path, body, http_status, expected):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(http_status, content=json.dumps(body), request=request)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(provenance, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(
        provenance.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    result = await provenance.verify_source("10.5555/example-paper")
    assert result["status"] == expected


@pytest.mark.anyio
async def test_cache_retraction_lookup_survives_crossref_outage(monkeypatch, tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"10.5555/retracted": "Known retraction."}), encoding="utf-8")
    monkeypatch.setattr(provenance, "CACHE_PATH", cache)

    async def unreachable(_: str):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(provenance, "_fetch_crossref", unreachable)
    result = await provenance.verify_source("https://doi.org/10.5555/retracted")
    assert result["status"] == "retracted"
    assert result["confidence"] == 0.98
