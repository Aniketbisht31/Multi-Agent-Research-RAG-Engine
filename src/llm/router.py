"""HTTPX-based router for OpenAI-compatible chat and embedding APIs."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

MAX_ATTEMPTS = 3


class Tier(str, Enum):
    REASONING = "REASONING"
    FAST = "FAST"
    TOOL = "TOOL"
    EMBED = "EMBED"


class LLMRouterError(RuntimeError):
    """Raised when every configured provider has failed."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    base_url: str
    model: str


def _env_name(tier: Tier, field: str, provider: str | None = None) -> str:
    prefix = tier.value if provider is None else f"{tier.value}_{provider.upper()}"
    return f"{prefix}_{field}"


def _config(tier: Tier, provider: str | None = None) -> ProviderConfig:
    """Read a tier config, optionally using provider-specific overrides.

    Cascades can define e.g. ``FAST_GROQ_API_KEY``, ``FAST_GROQ_BASE_URL``,
    and ``FAST_GROQ_MODEL``. Fields not overridden inherit the tier default.
    """
    def value(field: str) -> str:
        return os.getenv(_env_name(tier, field, provider)) or os.getenv(_env_name(tier, field), "")

    return ProviderConfig(provider or "primary", value("API_KEY"), value("BASE_URL").rstrip("/"), value("MODEL"))


def _validate(config: ProviderConfig) -> None:
    missing = [name for name, value in (("API_KEY", config.api_key), ("BASE_URL", config.base_url), ("MODEL", config.model)) if not value]
    if missing:
        raise LLMRouterError(f"{config.name}: missing {', '.join(missing)}")


def get_client(tier: Tier) -> httpx.AsyncClient:
    """Return an HTTPX client configured from the tier's primary environment variables."""
    config = _config(tier)
    _validate(config)
    return httpx.AsyncClient(
        base_url=config.base_url,
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        timeout=httpx.Timeout(30.0),
    )


def _providers(tier: Tier) -> list[str | None]:
    cascade = [item.strip().lower() for item in os.getenv(f"{tier.value}_CASCADE", "").split(",") if item.strip()]
    return [None, *cascade]


async def _post_with_retries(client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = await client.post(path, json=payload)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = httpx.HTTPStatusError(
                    f"retryable HTTP {response.status_code}", request=response.request, response=response
                )
            else:
                response.raise_for_status()
                return response.json()
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            last_error = error

        if attempt < MAX_ATTEMPTS - 1:
            await asyncio.sleep(2**attempt)
    raise LLMRouterError(f"failed after {MAX_ATTEMPTS} attempts: {last_error}")


async def _request(tier: Tier, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for provider in _providers(tier):
        config = _config(tier, provider)
        identity = (config.api_key, config.base_url, config.model)
        if identity in seen:
            continue
        seen.add(identity)
        try:
            _validate(config)
            async with httpx.AsyncClient(
                base_url=config.base_url,
                headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
                timeout=httpx.Timeout(30.0),
            ) as client:
                return await _post_with_retries(client, path, {"model": config.model, **payload})
        except (LLMRouterError, httpx.HTTPError) as error:
            failures.append(f"{config.name}: {error}")
    raise LLMRouterError(f"All {tier.value} providers exhausted. " + " | ".join(failures))


async def call_llm(tier: Tier, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint."""
    if tier is Tier.EMBED:
        raise ValueError("Use embed() for the EMBED tier.")
    return await _request(tier, "chat/completions", {"messages": messages, **kwargs})


async def embed(tier: Tier, texts: list[str]) -> list[list[float]]:
    """Embed text with an OpenAI-compatible ``/embeddings`` endpoint."""
    if tier is not Tier.EMBED:
        raise ValueError("embed() only accepts Tier.EMBED.")
    result = await _request(tier, "embeddings", {"input": texts})
    return [item["embedding"] for item in result["data"]]
