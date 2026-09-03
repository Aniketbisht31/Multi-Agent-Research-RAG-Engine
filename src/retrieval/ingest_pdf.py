"""CLI for extracting, chunking, and indexing a PDF."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pypdf import PdfReader

from .hybrid import chunk_text, ingest


async def ingest_pdf(path: Path) -> int:
    reader = PdfReader(path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunk_text(text)
    await ingest(chunks, [{"source": str(path), "chunk": index} for index in range(len(chunks))])
    print(f"Indexed {len(chunks)} chunks from {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    return asyncio.run(ingest_pdf(parser.parse_args().pdf))


if __name__ == "__main__":
    raise SystemExit(main())
