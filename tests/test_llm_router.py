import httpx
import pytest

from src.llm import router


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_call_llm_retries_a_rate_limit(monkeypatch):
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=request)

    monkeypatch.setenv("FAST_API_KEY", "test-key")
    monkeypatch.setenv("FAST_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("FAST_MODEL", "test-model")
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(router.asyncio, "sleep", lambda _: _no_sleep())
    monkeypatch.setattr(
        router.httpx,
        "AsyncClient",
        lambda **kwargs: real_async_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    result = await router.call_llm(router.Tier.FAST, [{"role": "user", "content": "hello"}])

    assert result["choices"][0]["message"]["content"] == "ok"
    assert attempts == 2


async def _no_sleep() -> None:
    return None
