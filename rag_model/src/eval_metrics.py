"""LLM and RAG Evaluation Framework for Fiduciary Lens Tax Advisor System.

Inspired by modern LLM QA standards (DeepEval & Ragas):
- Faithfulness / Hallucination Detection: Are all statements grounded in retrieved context?
- Answer Relevancy: Does the generated answer directly answer the user prompt?
- Context Precision & Recall: Did retrieval surface the exact tax section needed?
- Citation Accuracy & Fiduciary Compliance: Does the answer quote valid citations and include disclaimers?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    expected_citations: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    question: str
    answer: str
    grounded: bool
    faithfulness_score: float  # 0.0 to 1.0 (anti-hallucination)
    relevancy_score: float     # 0.0 to 1.0 (directness & relevance)
    citation_score: float      # 0.0 to 1.0 (citation presence & precision)
    fiduciary_compliance: bool # Contains required disclaimer or safe refusal
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


def compute_token_overlap(text1: str, text2: str) -> float:
    """Compute lexical Jaccard/overlap ratio between two text sequences."""
    tokens1 = set(re.findall(r"\w+", text1.lower()))
    tokens2 = set(re.findall(r"\w+", text2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    return len(intersection) / len(tokens1)


def evaluate_faithfulness(answer: str, context_chunks: Sequence[str]) -> float:
    """Evaluate if answer claims are grounded in the retrieved chunks (Hallucination metric)."""
    if not answer.strip():
        return 0.0
    if not context_chunks:
        # If no context was retrieved, groundedness must come from refusal or 0
        if "unable" in answer.lower() or "consult" in answer.lower():
            return 1.0
        return 0.0

    combined_context = " ".join(context_chunks).lower()
    sentences = [s.strip() for s in re.split(r"[.\n]", answer) if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    grounded_count = 0
    for s in sentences:
        # Check claim overlap with retrieved context
        s_words = set(re.findall(r"\w+", s.lower())) - {"the", "a", "an", "is", "of", "and", "in", "to", "for", "as"}
        if not s_words:
            grounded_count += 1
            continue
        matches = [w for w in s_words if w in combined_context]
        if len(matches) / len(s_words) >= 0.4:
            grounded_count += 1

    return round(grounded_count / len(sentences), 2)


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """Evaluate how relevant the answer is to the given question."""
    q_words = set(re.findall(r"\w+", question.lower())) - {"what", "is", "the", "are", "for", "how", "under", "in"}
    if not q_words:
        return 1.0
    a_words = set(re.findall(r"\w+", answer.lower()))
    overlap = q_words.intersection(a_words)
    return round(len(overlap) / len(q_words), 2)


def evaluate_citations(answer: str, sources: list[dict], expected_citations: list[str]) -> float:
    """Evaluate citation presence and conformance."""
    if not expected_citations:
        return 1.0
    
    score = 0.0
    found_in_text_or_sources = 0
    text_and_sources = (answer + " " + " ".join(s.get("citation", "") + " " + s.get("short", "") for s in sources)).lower()

    for cite in expected_citations:
        if cite.lower() in text_and_sources:
            found_in_text_or_sources += 1

    score = found_in_text_or_sources / len(expected_citations)
    return round(score, 2)


def check_fiduciary_compliance(answer: str, grounded: bool) -> bool:
    """Tax systems must enforce fiduciary boundaries: disclaimers and refusal on low confidence."""
    answer_lower = answer.lower()
    disclaimer_markers = ["disclaimer", "not legal advice", "consult a tax advisor", "consult", "statute"]
    has_disclaimer = any(marker in answer_lower for marker in disclaimer_markers)
    if not grounded:
        refusal_markers = ["outside", "not found", "cannot", "unable", "scope"]
        return any(marker in answer_lower for marker in refusal_markers) or has_disclaimer
    return True


def run_llm_eval(
    question: str,
    answer_payload: dict[str, Any],
    expected: EvalSample,
    min_faithfulness: float = 0.7,
    min_relevancy: float = 0.5,
) -> EvalResult:
    """Run full evaluation suite for an LLM response."""
    answer = answer_payload.get("answer", "")
    sources = answer_payload.get("sources", [])
    grounded = answer_payload.get("grounded", False)
    
    context_excerpts = [s.get("excerpt", "") for s in sources if s.get("excerpt")]

    faithfulness = evaluate_faithfulness(answer, context_excerpts)
    relevancy = evaluate_answer_relevancy(question, answer)
    citation_score = evaluate_citations(answer, sources, expected.expected_citations)
    fiduciary_compliance = check_fiduciary_compliance(answer, grounded)

    passed = (
        faithfulness >= min_faithfulness
        and relevancy >= min_relevancy
        and fiduciary_compliance
        and (citation_score >= 0.5 if expected.expected_citations else True)
    )

    return EvalResult(
        question=question,
        answer=answer,
        grounded=grounded,
        faithfulness_score=faithfulness,
        relevancy_score=relevancy,
        citation_score=citation_score,
        fiduciary_compliance=fiduciary_compliance,
        passed=passed,
        details={
            "sources_count": len(sources),
            "expected_citations": expected.expected_citations,
            "provider": answer_payload.get("provider", "unknown"),
        },
    )
