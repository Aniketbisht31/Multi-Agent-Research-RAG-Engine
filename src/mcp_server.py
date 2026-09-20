"""Verity's stdio MCP server."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from src.checks.critic import grade_evidence as _grade_evidence
from src.checks.originality import check_originality as _check_originality
from src.checks.provenance import verify_source as _verify_source
from src.pipeline.research import research as _research

mcp = MCPServer(
    "verity-mcp",
    description="Evidence validation tools for research agents.",
)


@mcp.tool()
async def verify_source(doi_or_url: str) -> dict[str, str | float | None]:
    """Check a paper DOI or doi.org URL for retraction or correction signals.

    Call this before relying on a scholarly source, especially when factual
    accuracy or a paper's publication status matters. It never raises for bad
    input or unavailable Crossref metadata; those return ``unverified``.
    """
    return await _verify_source(doi_or_url)


@mcp.tool()
async def grade_evidence(claim: str, source_text: str) -> dict[str, str]:
    """Strictly assess whether supplied source text supports a specific claim.

    Call this when an agent has a proposed claim and its cited excerpt, and
    needs an explicit supports/contradicts/unclear assessment rather than a
    plausibility judgment.
    """
    return (await _grade_evidence(claim, source_text)).model_dump()


@mcp.tool()
async def check_originality(draft_text: str, source_texts: list[str]) -> dict[str, Any]:
    """Detect sentence-level verbatim copying and close paraphrases in a draft.

    Call this before publishing or submitting a draft when it must be compared
    with supplied source material for attribution or originality review.
    """
    return await _check_originality(draft_text, source_texts)


@mcp.tool()
async def research(query: str, domain: str | None = None) -> dict[str, Any]:
    """Research a query across local, scholarly, and web evidence sources.

    Call this for a cited evidence-grounded answer. It uses bounded retrieval
    rewrites and fallbacks, then reports source citations and originality.
    """
    return await _research(query, domain)


def main() -> None:
    """Run the MCP server over stdio for Claude Desktop and Claude Code."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
