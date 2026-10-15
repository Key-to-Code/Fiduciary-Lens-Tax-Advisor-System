"""Pytest suite for LLM & RAG Evaluation.

Tests key criteria from the LLM QA Guide:
1. Faithfulness (Anti-Hallucination)
2. Answer Relevancy
3. Context Retrieval & Precision
4. Safe Refusal on Out-of-Domain queries
5. Citation Verification & Fiduciary Compliance
"""

from __future__ import annotations

import pytest
from rag_model.src.eval_metrics import (
    EvalSample,
    evaluate_answer_relevancy,
    evaluate_citations,
    evaluate_faithfulness,
    run_llm_eval,
)


@pytest.fixture
def golden_tax_dataset():
    return [
        EvalSample(
            question="What is the deduction limit for health insurance premiums under section 80D?",
            ground_truth="Section 80D allows deduction up to 25,000 for self and family, and additional 25,000 or 50,000 for parents.",
            expected_citations=["80D"],
            expected_keywords=["deduction", "health insurance", "premium"],
        ),
        EvalSample(
            question="Can a non-resident individual claim rebate under section 87A?",
            ground_truth="Rebate under Section 87A is available only to a resident individual.",
            expected_citations=["87A"],
            expected_keywords=["resident", "rebate"],
        ),
    ]


def test_faithfulness_metric_detects_hallucination():
    """Ensure hallucinated statements receive low faithfulness score."""
    context = ["Section 80C allows deductions up to 1.5 Lakhs for PPF and ELSS."]
    hallucinated_answer = "Section 80C provides a 100% tax credit on cryptocurrency trading in the Bahamas."
    
    score = evaluate_faithfulness(hallucinated_answer, context)
    assert score < 0.4, f"Expected low faithfulness score for hallucinated answer, got {score}"


def test_faithfulness_metric_rewards_grounded_answer():
    """Ensure answers grounded in retrieved context pass faithfulness check."""
    context = ["Under Section 80C, deduction up to 150000 rupees is allowable for investments in PPF and ELSS."]
    grounded_answer = "Under Section 80C, a deduction up to 150000 rupees is allowed for PPF and ELSS investments."
    
    score = evaluate_faithfulness(grounded_answer, context)
    assert score >= 0.7, f"Expected high faithfulness score for grounded answer, got {score}"


def test_relevancy_metric():
    """Ensure answer addresses the specific question asked."""
    question = "What is the capital gains tax on equity shares held for more than 12 months?"
    relevant_answer = "Long term capital gains on equity shares held for more than 12 months are taxed at 12.5%."
    irrelevant_answer = "You should register your company under the Companies Act 2013 with ROC."

    rel_score = evaluate_answer_relevancy(question, relevant_answer)
    irrel_score = evaluate_answer_relevancy(question, irrelevant_answer)

    assert rel_score > irrel_score
    assert rel_score >= 0.5


def test_eval_suite_on_tax_qa_golden_sample(golden_tax_dataset):
    """End-to-end evaluation runner verifying payload against a golden test case."""
    sample = golden_tax_dataset[0]
    
    mock_payload = {
        "answer": "Under Section 80D of the Income-tax Act, a deduction is available for health insurance premiums.",
        "grounded": True,
        "provider": "extractive",
        "sources": [
            {
                "citation": "Income-tax Act, Section 80D",
                "short": "Act s.80D",
                "excerpt": "Deduction in respect of health insurance premia shall be allowed subject to conditions of Section 80D.",
            }
        ],
    }

    result = run_llm_eval(sample.question, mock_payload, sample)
    assert result.passed is True
    assert result.faithfulness_score >= 0.7
    assert result.relevancy_score >= 0.5
    assert result.fiduciary_compliance is True
    assert result.citation_score >= 0.5


def test_eval_suite_flags_ungrounded_unrefused_response():
    """If an answer is ungrounded without standard fiduciary refusal, it must fail eval."""
    sample = EvalSample(
        question="How to cook pasta?",
        ground_truth="Not applicable to tax statutes.",
        expected_citations=[],
    )

    bad_payload = {
        "answer": "Boil water and add salt, then cook pasta for 8 minutes.",
        "grounded": False,
        "sources": [],
        "provider": "openai",
    }

    result = run_llm_eval(sample.question, bad_payload, sample)
    assert result.passed is False
