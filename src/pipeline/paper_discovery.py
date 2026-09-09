"""Scholarly fallback searches using arXiv and Semantic Scholar."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any

import httpx


async def discover_papers(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Search arXiv and Semantic Scholar and return abstracts as source records."""
    headers = {"User-Agent": "verity-mcp/0.1"}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
        arxiv_task = client.get("https://export.arxiv.org/api/query", params={"search_query": f"all:{query}", "start": 0, "max_results": limit})
        semantic_task = client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": limit, "fields": "title,abstract,url,externalIds"},
        )
        arxiv_response, semantic_response = await __import__("asyncio").gather(arxiv_task, semantic_task, return_exceptions=True)

    records: list[dict[str, Any]] = []
    if isinstance(arxiv_response, httpx.Response) and arxiv_response.is_success:
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(arxiv_response.text)
            for entry in root.findall("atom:entry", namespace):
                summary = (entry.findtext("atom:summary", default="", namespaces=namespace) or "").strip()
                if summary:
                    records.append({"title": entry.findtext("atom:title", default="arXiv result", namespaces=namespace).strip(), "text": summary, "url": entry.findtext("atom:id", default="", namespaces=namespace), "provider": "arXiv"})
        except ET.ParseError:
            pass
    if isinstance(semantic_response, httpx.Response) and semantic_response.is_success:
        try:
            for paper in semantic_response.json().get("data", []):
                if paper.get("abstract"):
                    records.append({"title": paper.get("title") or "Semantic Scholar result", "text": paper["abstract"], "url": paper.get("url") or "", "provider": "Semantic Scholar"})
        except (ValueError, AttributeError):
            pass
    return records[: limit * 2]
