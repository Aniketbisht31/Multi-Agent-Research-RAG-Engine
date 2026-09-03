"""Persistent Chroma retrieval with BM25 and reciprocal-rank fusion."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from src.llm import Tier, embed

COLLECTION_NAME = "verity_chunks"
CHROMA_PATH = Path("data/chroma")
RRF_K = 60

_collection: Any | None = None


def _get_collection() -> Any:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text near paragraph boundaries, preserving a character overlap."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + 2 + len(paragraph) <= chunk_size:
            current = f"{current}\n\n{paragraph}"
            continue
        if current:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
        while len(paragraph) > chunk_size:
            prefix = f"{current}\n\n" if current else ""
            available = chunk_size - len(prefix)
            cut = paragraph.rfind(" ", 0, available)
            cut = cut if cut > available // 2 else available
            chunks.append(f"{prefix}{paragraph[:cut]}".strip())
            paragraph = paragraph[max(0, cut - overlap) :].lstrip()
            current = ""
        current = f"{current}\n\n{paragraph}".strip() if current else paragraph
    if current:
        chunks.append(current)
    return chunks


async def ingest(chunks: list[str], metadata: list[dict[str, Any]]) -> list[str]:
    """Embed and persist chunks in the local Chroma collection."""
    if len(chunks) != len(metadata):
        raise ValueError("chunks and metadata must have the same length")
    if not chunks:
        return []
    ids = [str(uuid.uuid4()) for _ in chunks]
    vectors = await embed(Tier.EMBED, chunks)
    _get_collection().add(ids=ids, documents=chunks, metadatas=metadata, embeddings=vectors)
    return ids


def _all_documents() -> tuple[list[str], list[str], list[dict[str, Any]]]:
    data = _get_collection().get(include=["documents", "metadatas"])
    return data["ids"], data["documents"], data["metadatas"]


async def bm25_search(query: str, k: int) -> list[dict[str, Any]]:
    """Run BM25 against documents currently stored in Chroma."""
    ids, documents, metadatas = _all_documents()
    if not documents:
        return []
    scores = BM25Okapi([_tokens(document) for document in documents]).get_scores(_tokens(query))
    ranked = sorted(range(len(documents)), key=lambda index: scores[index], reverse=True)[:k]
    return [
        {"id": ids[index], "document": documents[index], "metadata": metadatas[index], "score": float(scores[index])}
        for index in ranked
    ]


async def dense_search(query: str, k: int) -> list[dict[str, Any]]:
    """Embed a query and retrieve nearest Chroma vectors."""
    if k <= 0 or _get_collection().count() == 0:
        return []
    vector = (await embed(Tier.EMBED, [query]))[0]
    result = _get_collection().query(query_embeddings=[vector], n_results=k, include=["documents", "metadatas", "distances"])
    return [
        {"id": item_id, "document": document, "metadata": metadata, "score": -float(distance)}
        for item_id, document, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


async def hybrid_search(query: str, k: int) -> list[dict[str, Any]]:
    """Fuse dense and lexical rankings with Reciprocal Rank Fusion."""
    if k <= 0:
        return []
    bm25_results, dense_results = await bm25_search(query, k), await dense_search(query, k)
    fused: dict[str, dict[str, Any]] = {}
    for results in (bm25_results, dense_results):
        for rank, item in enumerate(results, start=1):
            entry = fused.setdefault(item["id"], {**item, "score": 0.0})
            entry["score"] += 1 / (RRF_K + rank)
    return sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:k]
