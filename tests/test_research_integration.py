import os

import pytest

from src.pipeline.research import research


@pytest.mark.integration
@pytest.mark.anyio
async def test_research_returns_cited_answer_against_live_services():
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("Set RUN_INTEGRATION=1 and configure provider API keys to run live research.")
    result = await research("What is the chemical symbol for water?")
    assert result["citations"]
    assert result["confidence"] in {"high", "medium", "low"}
