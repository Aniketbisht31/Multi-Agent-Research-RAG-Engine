# Verity MCP

Verity is an evidence-focused Model Context Protocol (MCP) server for research agents. It combines local hybrid retrieval, scholarly and web fallbacks, source provenance checks, evidence grading, and originality checks to produce cited answers.

## Features

- Hybrid local retrieval using Chroma, BM25, and reciprocal-rank fusion.
- Configurable OpenAI-compatible LLM and embedding providers with retry and cascade support.
- Evidence-first research workflow that tries local, scholarly, and web sources in sequence.
- DOI provenance checks for retraction and correction signals through Crossref.
- Claim-to-source grading and sentence-level originality checks.
- MCP tools: `research`, `verify_source`, `grade_evidence`, and `check_originality`.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- API credentials for the configured chat and embedding providers

## Installation

```bash
git clone https://github.com/Aniketbisht31/Multi-Agent-Research-RAG-Engine.git
cd Multi-Agent-Research-RAG-Engine
uv sync
cp .env.example .env
```

Configure the `REASONING_*`, `FAST_*`, and `EMBED_*` settings in `.env`. Providers must expose OpenAI-compatible `/chat/completions` and `/embeddings` endpoints. Optional `*_CASCADE` settings provide fallback providers.

Check the configuration without exposing secrets:

```bash
uv run python -m src.llm.status
```

## Run locally

Index a PDF into the local Chroma store:

```bash
uv run python -m src.retrieval.ingest_pdf path/to/paper.pdf
```

Run the sample end-to-end demonstration:

```bash
uv run python scripts/demo.py
```

Start the MCP server over stdio:

```bash
uv run python -m src.mcp_server
```

For Claude Desktop configuration, see [CONNECTING.md](CONNECTING.md).

## MCP tools

| Tool | Purpose |
| --- | --- |
| `research(query, domain?)` | Produces an evidence-grounded response with citations and originality report. |
| `verify_source(doi_or_url)` | Checks DOI metadata and local cache for retraction or correction signals. |
| `grade_evidence(claim, source_text)` | Returns whether a source supports, contradicts, or leaves a claim unclear. |
| `check_originality(draft_text, source_texts)` | Flags likely copied or close-paraphrased sentences. |

## Tests

```bash
uv run pytest -q
```

The live research integration test is skipped by default. Enable it only after configuring providers:

```bash
RUN_INTEGRATION=1 uv run pytest -m integration
```

## Project layout

```text
src/
  checks/       Evidence grading, originality, and DOI provenance
  llm/          Provider routing and status CLI
  pipeline/     Planning, retrieval fallbacks, and answer generation
  retrieval/    PDF ingestion and hybrid vector/BM25 search
  mcp_server.py MCP tool definitions
tests/          Unit and opt-in integration tests
samples/        Small local corpus used by the demo
```
