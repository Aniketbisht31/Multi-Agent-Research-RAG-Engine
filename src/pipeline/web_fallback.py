"""Web fallback via Tavily or DuckDuckGo's instant-answer endpoint."""

from __future__ import annotations

import os
from typing import Any

import httpx


async def search_web(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Return web snippets, preferring Tavily when its API key is configured."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        key = os.getenv("TAVILY_API_KEY")
        if key:
            try:
                response = await client.post("https://api.tavily.com/search", headers={"Authorization": f"Bearer {key}"}, json={"query": query, "max_results": limit, "include_answer": False})
                response.raise_for_status()
                return [{"title": item.get("title") or "Web result", "text": item.get("content") or "", "url": item.get("url") or "", "provider": "Tavily"} for item in response.json().get("results", []) if item.get("content")]
            except (httpx.HTTPError, ValueError):
                pass
        try:
            response = await client.get("https://api.duckduckgo.com/", params={"q": query, "format": "json", "no_html": 1})
            response.raise_for_status()
            payload = response.json()
            records = []
            if payload.get("AbstractText"):
                records.append({"title": payload.get("Heading") or "DuckDuckGo result", "text": payload["AbstractText"], "url": payload.get("AbstractURL") or "", "provider": "DuckDuckGo"})
            for topic in payload.get("RelatedTopics", []):
                if topic.get("Text"):
                    records.append({"title": "DuckDuckGo related result", "text": topic["Text"], "url": topic.get("FirstURL") or "", "provider": "DuckDuckGo"})
            return records[:limit]
        except (httpx.HTTPError, ValueError):
            return []
