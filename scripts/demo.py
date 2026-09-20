"""Run a readable end-to-end Verity MCP demonstration.

Run from the project root after copying .env.example to .env and supplying
the EMBED, FAST, and REASONING provider credentials:
    uv run python scripts/demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checks.originality import check_originality
from src.checks.provenance import verify_source
from src.pipeline.research import research
from src.retrieval.hybrid import chunk_text, ingest

SAMPLES = ROOT / "samples"


def heading(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def load_samples() -> int:
    chunks: list[str] = []
    metadata = []
    for path in sorted(SAMPLES.glob("*.md")):
        document_chunks = chunk_text(path.read_text(encoding="utf-8"))
        chunks.extend(document_chunks)
        metadata.extend({"title": path.stem.replace("_", " ").title(), "url": path.as_uri(), "source": str(path)} for _ in document_chunks)
    await ingest(chunks, metadata)
    return len(chunks)


def show_provenance(label: str, result: dict) -> None:
    print(f"{label:<12} {result['status']:<11} confidence {result['confidence']:.0%}")
    if result.get("caveat"):
        print(f"{'':12} {result['caveat']}")


async def main() -> None:
    heading("1. Loading the sample corpus")
    count = await load_samples()
    print(f"Indexed {count} chunks from {SAMPLES.name}/.")

    heading("2. Research answer")
    result = await research("What powers the Northstar field station, and how long can its essential instruments operate without sunlight?")
    print(result["answer"])
    print(f"\nConfidence: {result['confidence']}")
    print("Sources:")
    for citation in result["citations"]:
        location = f" — {citation['url']}" if citation["url"] else ""
        print(f"  [{citation['id']}] {citation['title']} ({citation['provider']}){location}")
    report = result["originality_report"]
    print(f"Originality score: {report['score'] if report['score'] is not None else 'unavailable'}")

    heading("3. Source provenance")
    print("DOI          status      confidence")
    show_provenance("Retracted", await verify_source("10.1126/science.290.5493.963"))
    show_provenance("Clean", await verify_source("10.1038/nature12373"))

    heading("4. Originality comparison")
    source = "The station battery reserve can operate essential instruments for 36 hours without sunlight."
    copied = "The station battery reserve can operate essential instruments for 36 hours without sunlight."
    original = "The field team planted willow saplings beside the stream after lunch."
    copied_report = await check_originality(copied, [source])
    original_report = await check_originality(original, [source])
    print(f"Copied paragraph:   score {copied_report['score']:.2f}; flags: {', '.join(flag['type'] for flag in copied_report['flags']) or 'none'}")
    print(f"Original paragraph: score {original_report['score']:.2f}; flags: {', '.join(flag['type'] for flag in original_report['flags']) or 'none'}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        print(f"\nDemo could not complete: {error}")
        print("Copy .env.example to .env and configure EMBED, FAST, and REASONING provider credentials.")
