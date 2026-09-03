import chromadb
import pytest

from src.retrieval import hybrid


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_hybrid_search_finds_obvious_document(monkeypatch):
    documents = [
        "Oranges are citrus fruit.",
        "Python is a programming language.",
        "Apples are sweet orchard fruit.",
        "Saturn has distinctive rings.",
        "A bicycle has two wheels.",
        "Rain falls from clouds.",
    ]

    async def fake_embed(_, texts):
        return [[1.0, 0.0] if "apple" in text.lower() else [0.0, 1.0] for text in texts]

    client = chromadb.EphemeralClient()
    monkeypatch.setattr(hybrid, "_collection", client.get_or_create_collection("test_chunks"))
    monkeypatch.setattr(hybrid, "embed", fake_embed)
    await hybrid.ingest(documents, [{"source": "fake"} for _ in documents])

    results = await hybrid.hybrid_search("Which orchard fruit is sweet? Tell me about apples.", k=3)

    assert results[0]["document"] == "Apples are sweet orchard fruit."
