"""Retraction and correction checks backed by Crossref and a local fallback."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

CACHE_PATH = Path("data/retraction_cache.json")
DEFAULT_RETRACTED = {
    "10.1126/science.290.5493.963": "Retracted: A light-emitting field-effect transistor (Schön et al.).",
    "10.1126/science.288.5466.656": "Retracted: A superconducting field-effect switch (Schön et al.).",
}
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/[-._;()/:a-z0-9]+$", re.IGNORECASE)


def _normalise_doi(doi_or_url: str) -> str | None:
    value = doi_or_url.strip()
    value = re.sub(r"^doi:\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    value = value.rstrip(".").lower()
    return value if DOI_PATTERN.fullmatch(value) else None


def _load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(DEFAULT_RETRACTED, indent=2) + "\n", encoding="utf-8")
        return DEFAULT_RETRACTED.copy()
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return {str(doi).lower(): str(note) for doi, note in raw.items()}
    except (OSError, json.JSONDecodeError, AttributeError):
        return DEFAULT_RETRACTED.copy()


async def _fetch_crossref(doi: str) -> dict[str, Any] | None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0), headers={"User-Agent": "verity-mcp/0.1 (provenance checker)"}) as client:
        response = await client.get(f"https://api.crossref.org/works/{quote(doi, safe='')}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        return payload.get("message") if isinstance(payload, dict) else None


def _crossref_status(metadata: dict[str, Any]) -> str:
    """Interpret Crossref relation/update fields without treating missing data as clean evidence."""
    relation_text = json.dumps(metadata.get("relation", {})).lower()
    update_text = json.dumps(metadata.get("update-to", [])).lower()
    signals = f"{relation_text} {update_text}"
    if "retract" in signals:
        return "retracted"
    if "correct" in signals or "erratum" in signals:
        return "corrected"
    return "clean"


async def verify_source(doi_or_url: str) -> dict[str, str | float | None]:
    """Return a safe provenance status; network and input failures are unverified."""
    doi = _normalise_doi(doi_or_url)
    if not doi:
        return {"status": "unverified", "confidence": 0.0, "caveat": "Input is not a valid DOI or doi.org URL."}

    cached = _load_cache()
    if doi in cached:
        return {"status": "retracted", "confidence": 0.98, "caveat": f"Local retraction cache: {cached[doi]}"}

    try:
        metadata = await _fetch_crossref(doi)
    except (httpx.HTTPError, ValueError, json.JSONDecodeError):
        return {"status": "unverified", "confidence": 0.0, "caveat": "Crossref could not be reached or returned invalid metadata."}
    if not metadata:
        return {"status": "unverified", "confidence": 0.0, "caveat": "DOI was not found in Crossref."}

    status = _crossref_status(metadata)
    if status == "retracted":
        return {"status": status, "confidence": 0.9, "caveat": "Crossref metadata contains a retraction relation."}
    if status == "corrected":
        return {"status": status, "confidence": 0.8, "caveat": "Crossref metadata contains a correction relation."}
    return {"status": "clean", "confidence": 0.6, "caveat": "No Crossref retraction/correction signal found; such signals may be incomplete."}
