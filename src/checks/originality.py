"""Sentence-level originality checks using lexical and semantic matching."""

from __future__ import annotations

import math
import re
from typing import Any

from src.llm import Tier, embed

NGRAM_SIZE = 8
VERBATIM_THRESHOLD = 0.8
PARAPHRASE_THRESHOLD = 0.85


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence.strip()]


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(text: str) -> set[tuple[str, ...]]:
    tokens = _tokens(text)
    return {tuple(tokens[index : index + NGRAM_SIZE]) for index in range(len(tokens) - NGRAM_SIZE + 1)}


def _ngram_overlap(span: str, source: str) -> float:
    span_grams = _ngrams(span)
    if not span_grams:
        return 0.0
    return len(span_grams & _ngrams(source)) / len(span_grams)


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


async def check_originality(draft_text: str, source_texts: list[str]) -> dict[str, float | list[dict[str, Any]]]:
    """Find copied or closely paraphrased draft sentences against supplied sources."""
    draft_sentences = _sentences(draft_text)
    source_sentences = [_sentences(source) for source in source_texts]
    flat_sources = [sentence for sentences in source_sentences for sentence in sentences]
    if not draft_sentences or not flat_sources:
        return {"score": 1.0, "flags": []}

    vectors = await embed(Tier.EMBED, [*draft_sentences, *flat_sources])
    draft_vectors = vectors[: len(draft_sentences)]
    source_vectors = vectors[len(draft_sentences) :]
    source_vector_groups: list[list[list[float]]] = []
    cursor = 0
    for sentences in source_sentences:
        source_vector_groups.append(source_vectors[cursor : cursor + len(sentences)])
        cursor += len(sentences)

    flags: list[dict[str, Any]] = []
    for span, vector in zip(draft_sentences, draft_vectors):
        overlaps = [_ngram_overlap(span, source) for source in source_texts]
        source_index = max(range(len(source_texts)), key=overlaps.__getitem__)
        if overlaps[source_index] > VERBATIM_THRESHOLD:
            flags.append({"span": span, "type": "verbatim", "matched_source": source_index})
            continue

        similarities = [max((_cosine(vector, candidate) for candidate in group), default=0.0) for group in source_vector_groups]
        source_index = max(range(len(source_texts)), key=similarities.__getitem__)
        if similarities[source_index] > PARAPHRASE_THRESHOLD:
            flags.append({"span": span, "type": "paraphrase", "matched_source": source_index})

    return {"score": max(0.0, 1.0 - len(flags) / len(draft_sentences)), "flags": flags}
