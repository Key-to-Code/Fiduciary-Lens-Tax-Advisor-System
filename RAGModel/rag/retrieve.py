"""Hybrid retrieval: dense cosine + BM25, fused, then deduplicated by provision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config, embedder
from .index import SearchIndex
from .kb import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float        # fused rank score, 0..1 — for ordering only
    dense_score: float  # raw cosine similarity — comparable across queries
    lexical_score: float


def _normalise(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    low, high = float(scores.min()), float(scores.max())
    if high - low < 1e-9:
        return np.ones_like(scores)
    return (scores - low) / (high - low)


def _statute_weight(chunk: Chunk) -> float:
    """Demote rate tables belonging to a superseded Act.

    Only Finance Act passages carry `applies_to`, and only those that set rates
    under an Act other than the principal one are penalised. A passage that
    straddles both sub-parts names the principal year too, so it is not demoted.
    """
    if chunk.applies_to and config.PRINCIPAL_ACT_YEAR not in chunk.applies_to:
        return config.SUPERSEDED_PENALTY
    return 1.0


def search(index: SearchIndex, question: str, top_k: int | None = None,
           candidate_k: int | None = None) -> list[Hit]:
    top_k = top_k or config.TOP_K
    candidate_k = candidate_k or config.CANDIDATE_K

    query_vector = embedder.encode_query(question)
    dense_scores, dense_ids = index.dense.search(query_vector, candidate_k)
    dense_scores, dense_ids = dense_scores[0], dense_ids[0]

    lexical_ids, lexical_scores = index.lexical.search(question, candidate_k)

    dense_by_id = dict(zip(dense_ids.tolist(), dense_scores.tolist()))
    lexical_by_id = dict(zip(lexical_ids.tolist(), lexical_scores.tolist()))

    # Min-max normalise within each candidate pool so the two scales are comparable,
    # then take a weighted sum. Documents missing from a pool score 0 there.
    dense_norm = dict(zip(dense_ids.tolist(), _normalise(dense_scores).tolist()))
    lexical_norm = dict(zip(lexical_ids.tolist(), _normalise(lexical_scores).tolist()))

    weight = config.DENSE_WEIGHT
    fused = {
        doc_id: (weight * dense_norm.get(doc_id, 0.0)
                 + (1 - weight) * lexical_norm.get(doc_id, 0.0))
        * _statute_weight(index.chunks[doc_id])
        for doc_id in set(dense_norm) | set(lexical_norm)
    }

    ranked = sorted(fused.items(), key=lambda item: -item[1])

    hits: list[Hit] = []
    seen_provisions: set[tuple[str, str | None]] = set()
    for doc_id, score in ranked:
        chunk = index.chunks[doc_id]
        # Adjacent chunks of one long section say much the same thing; spending
        # the context budget on distinct provisions retrieves more law per token.
        key = (chunk.document_name, chunk.number)
        if key in seen_provisions:
            continue
        seen_provisions.add(key)
        hits.append(Hit(
            chunk=chunk,
            score=round(float(score), 4),
            dense_score=round(float(dense_by_id.get(doc_id, 0.0)), 4),
            lexical_score=round(float(lexical_by_id.get(doc_id, 0.0)), 4),
        ))
        if len(hits) >= top_k:
            break

    return hits


def is_grounded(hits: list[Hit]) -> bool:
    """Whether retrieval found anything worth answering from.

    Gated on raw cosine, not the fused score: the fused score is normalised per
    query, so it reads 1.0 for the best of a uniformly irrelevant pool.
    """
    return bool(hits) and max(hit.dense_score for hit in hits) >= config.MIN_SCORE
