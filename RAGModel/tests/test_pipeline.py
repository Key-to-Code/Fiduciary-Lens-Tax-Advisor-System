"""Unit tests for the parts of the pipeline that must not silently regress.

Retrieval quality itself is measured by `eval_retrieval.py`; these cover the
metadata and guardrail logic that decides what a citation says and when the bot
refuses.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import json

import pytest

from rag import prompt, retrieve
from rag.kb import Chunk, _sort_key, load_chunks
from rag.lexical import BM25Index


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return load_chunks()


# --- citation metadata ---------------------------------------------------

def test_every_chunk_resolves_to_a_provision(chunks):
    """A passage with no provision cannot be cited, which means it cannot be used.

    Only a document's opening preamble may be unresolved -- it sits before the
    first numbered provision, so there is nothing to attribute it to.
    """
    first_ordinal = {}
    for chunk in chunks:
        first_ordinal.setdefault(chunk.document_name, chunk.ordinal)
    unresolved = [c.chunk_id for c in chunks
                  if not c.number and c.ordinal > first_ordinal[c.document_name]]
    assert not unresolved, f"{len(unresolved)} chunks lack a provision label"


def test_provision_numbers_ascend_within_the_act(chunks):
    """The forward-fill must not walk backwards; a drop means a false positive.

    Numbers are compared with the module's own key because sections carry letter
    suffixes ("354A" sorts after "354", not as an integer).
    """
    sections = [_sort_key(c.number) for c in chunks
                if c.doc_title.startswith("Income-tax Act") and c.unit == "Section"]
    assert sections == sorted(sections)


def test_known_headings_are_attached(chunks):
    by_provision = {(c.unit, c.number): c.heading for c in chunks if c.heading}
    assert "Residence in India" in by_provision[("Section", "6")]
    assert "Instalments of advance tax" in by_provision[("Section", "408")]


def test_schedules_are_not_labelled_as_sections(chunks):
    """Schedule content restarts numbering, so it must not inherit the last section."""
    schedule_xv = [c for c in chunks if c.unit == "Schedule" and c.number == "XV"]
    assert schedule_xv
    assert all(c.doc_title.startswith("Income-tax Act") for c in schedule_xv)


def test_citation_is_human_readable(chunks):
    section_6 = next(c for c in chunks if c.unit == "Section" and c.number == "6")
    assert section_6.citation.startswith("Income-tax Act, 2025, Section 6")
    assert section_6.short_citation == "Act s.6"


# --- lexical retrieval ---------------------------------------------------

def test_bm25_matches_provision_numbers():
    """The tokenizer must keep '80CCD'-style tokens whole, not split on the digits."""
    index = BM25Index.build([
        "Deduction under section 80CCD for pension contributions.",
        "Rates of depreciation on plant and machinery.",
        "Form No. 154 must be furnished to the prescribed authority.",
    ])
    ids, scores = index.search("80CCD", k=3)
    assert ids[0] == 0 and scores[0] > 0

    ids, _ = index.search("Form No. 154", k=3)
    assert ids[0] == 2


def test_bm25_returns_nothing_for_unknown_terms():
    index = BM25Index.build(["income tax act", "rules of assessment"])
    ids, scores = index.search("zzzznonexistent", k=3)
    assert len(ids) == 0 and len(scores) == 0


# --- guardrails ----------------------------------------------------------

@pytest.mark.parametrize("question", [
    "Should I invest in ELSS to save tax?",
    "How much tax do I owe on 12 lakhs?",
    "Am I eligible for the education loan deduction?",
    "Help me reduce my tax liability",
])
def test_personal_advice_is_flagged(question):
    assert prompt.is_personal_advice(question)


@pytest.mark.parametrize("question", [
    "What is the due date for advance tax?",
    "How is a perquisite defined?",
    "What deductions are allowed from house property income?",
])
def test_general_questions_are_not_flagged(question):
    assert not prompt.is_personal_advice(question)


def test_personal_advice_adds_a_steer_to_the_system_prompt():
    hits = []
    general, _ = prompt.build_messages("What is a perquisite?", hits)
    personal, _ = prompt.build_messages("Should I claim this perquisite?", hits)
    assert len(personal) > len(general)
    assert "Chartered Accountant" in personal


def test_context_is_numbered_and_carries_citations():
    class FakeChunk:
        content = "The text of the provision."
        citation = "Income-tax Act, 2025, Section 19 (Deductions from salaries)"
        short_citation = "Act s.19"
        document_name = "act.pdf"
        source_note = "in force from 1 April 2026"

    context = prompt.format_context([retrieve.Hit(FakeChunk(), 0.9, 0.8, 5.0)])
    assert context.startswith("[1] Income-tax Act, 2025, Section 19")
    assert "Act s.19" in context
    assert "in force from 1 April 2026" in context


def test_refusal_names_the_knowledge_base_scope():
    """A refusal that does not say what IS covered just frustrates the user."""
    assert "Income-tax Act, 2025" in prompt.REFUSAL
    assert "Income-tax Rules, 2026" in prompt.REFUSAL
    assert "Chartered Accountant" in prompt.REFUSAL


def test_disclaimer_states_it_is_not_advice():
    assert "not professional tax" in prompt.DISCLAIMER
    assert "Chartered Accountant" in prompt.DISCLAIMER


# --- grounding gate ------------------------------------------------------

def _hit(cosine: float) -> retrieve.Hit:
    return retrieve.Hit(chunk=None, score=1.0, dense_score=cosine, lexical_score=0.0)


def test_grounding_gate_uses_absolute_cosine_not_fused_score():
    """Fused scores are normalised per query, so they read 1.0 even for junk."""
    assert not retrieve.is_grounded([_hit(0.51), _hit(0.49)])
    assert retrieve.is_grounded([_hit(0.72), _hit(0.51)])


def test_empty_retrieval_is_not_grounded():
    assert not retrieve.is_grounded([])


# --- known coverage gaps -------------------------------------------------

@pytest.mark.parametrize("question", [
    "What are the income tax slabs for individuals?",
    "Tell me the tax slab rates",
    "What are the rates of income-tax this year?",
    "Should I choose the new regime or the old regime?",
    "How much tax on a salary of 18 lakhs?",
    "Calculate my tax liability",
    "My salary is 18 lakhs. How much tax should I pay?",
    "How much tax do I owe?",
    "What is my tax liability on Rs. 950000?",
])
def test_rate_questions_are_refused_as_uncovered(question):
    """Without a Finance Act in the KB, slab answers could only be invented."""
    refusal = prompt.uncovered_topic(question)
    assert refusal is not None
    assert "Finance Act" in refusal


@pytest.mark.parametrize("question", [
    "What are the income tax slabs for individuals?",
    "How much tax on a salary of 18 lakhs?",
])
def test_rate_guard_retires_once_a_finance_act_is_indexed(question):
    """Adding the Finance Act must retire the guard without a code edit."""
    assert prompt.uncovered_topic(question, rates_available=True) is None


@pytest.mark.parametrize("question", [
    "What deductions are allowed from house property income?",
    "What is the due date for advance tax?",
    "Who is required to deduct tax at source on rent?",
    "What is a perquisite?",
    "Is there a rebate for small taxpayers?",
    # Impersonal rate questions stay answerable: TDS rates ARE in the Act.
    "How much tax is deducted at source on rent?",
    "How is the tax payable by a company computed under the Act?",
])
def test_answerable_questions_are_not_treated_as_uncovered(question):
    assert prompt.uncovered_topic(question) is None


# --- Finance Act structure -----------------------------------------------
#
# The First Schedule carries rates for BOTH the Income-tax Act, 1961 (sub-part A)
# and the Income-tax Act, 2025 (sub-part B). Attributing a passage to the wrong
# one produces a citation that looks authoritative and is wrong, so the sub-part
# resolution is pinned here.

def _finance_kb(tmp_path, blocks):
    payload = [{"document_name": "Finance-Act-2026.pdf",
                "chunk_id": f"fa_{i}", "content": text, "metadata": {}}
               for i, text in enumerate(blocks)]
    path = tmp_path / "fa_kb.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_chunks(path)


def test_finance_act_filename_does_not_match_the_income_tax_act(tmp_path):
    """'Finance-Act-2026.pdf' contains 'act'; it must not inherit that profile."""
    chunks = _finance_kb(tmp_path, ["1. (1) This Act may be called the Finance Act, 2026."])
    assert chunks[0].doc_title == "Finance Act, 2026"


def test_finance_act_schedule_part_and_subpart_resolve(tmp_path):
    chunks = _finance_kb(tmp_path, [
        "THE FIRST SCHEDULE\n(See sections 2 and 3)\nPART I\n"
        "A.--INCOME-TAX UNDER THE INCOME-TAX ACT, 1961\nParagraph A\n"
        "(I) In the case of every individual...",
        "PART III\nRATES FOR CHARGING INCOME-TAX IN CERTAIN CASES\n"
        "under section 316(5) of the Income-tax Act, 2025\n"
        "B.--INCOME-TAX UNDER THE INCOME-TAX ACT, 2025\nParagraph A\n"
        "(1) where the total income does not exceed 250000 Nil;",
    ])
    first, third = chunks
    assert first.citation.startswith("Finance Act, 2026, First Schedule, Part I, Paragraph A")
    assert "Income-tax Act, 1961" in first.applies_to
    assert third.subdivision.startswith("Part III")
    assert third.applies_to == "Income-tax Act, 2025"


def test_part_numbers_in_the_table_of_contents_are_ignored(tmp_path):
    """PART I..IV also appear in the contents list, before any schedule opens."""
    chunks = _finance_kb(tmp_path, [
        "CONTENTS\nPART I\nPART II\nPART III\nPART IV\n",
        "THE FIRST SCHEDULE\nPART II\nRATES FOR DEDUCTION OF TAX AT SOURCE\n"
        "B.--INCOME-TAX UNDER THE INCOME-TAX ACT, 2025\n",
    ])
    assert chunks[0].subdivision is None
    assert chunks[1].subdivision == "Part II"


def test_a_chunk_straddling_the_subpart_switch_is_flagged(tmp_path):
    """Labelling a 1961/2025 straddle with one Act alone would be confidently wrong."""
    tail = "x" * 500  # push the switch past the "chunk starts here" window
    chunks = _finance_kb(tmp_path, [
        "THE FIRST SCHEDULE\nPART I\nA.--INCOME-TAX UNDER THE INCOME-TAX ACT, 1961\n",
        f"Paragraph E rates continue {tail}\n"
        "B.--INCOME-TAX UNDER THE INCOME-TAX ACT, 2025\nParagraph A\n",
    ])
    assert chunks[1].applies_to is not None
    assert "spans both" in chunks[1].applies_to
    assert "1961" in chunks[1].applies_to and "2025" in chunks[1].applies_to


def test_finance_act_short_citation_reads_naturally(tmp_path):
    chunks = _finance_kb(tmp_path, [
        "THE FIRST SCHEDULE\nPART III\nB.--INCOME-TAX UNDER THE INCOME-TAX ACT, 2025\n",
        "2. (1) Income-tax shall be charged at the rates specified.",
    ])
    assert chunks[0].short_citation == "FA 2026 First Sch, Part III"


# --- forward-fill robustness ---------------------------------------------

def _statute_kb(tmp_path, blocks):
    payload = [{"document_name": "Income-tax-Act-2025.pdf",
                "chunk_id": f"act_{i}", "content": text, "metadata": {}}
               for i, text in enumerate(blocks)]
    path = tmp_path / "act_kb.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_chunks(path)


def test_amended_provisions_in_brackets_are_detected(tmp_path):
    """Substituted text renders as '169.\n[(1) ...'; the bracket must not hide it."""
    chunks = _statute_kb(tmp_path, [
        "Effect to advance pricing agreement.\n169. \n[(1) Irrespective of anything...",
    ])
    assert chunks[0].number == "169"


def test_a_wrapped_cross_reference_does_not_hijack_the_forward_fill(tmp_path):
    """'...furnished under section\n263.' sits at a line start but is not section 263.

    Before the jump guard, one such match at section 168 made every genuine
    section from 169 to 262 inherit the citation for 263.
    """
    chunks = _statute_kb(tmp_path, [
        "Advance pricing agreement.\n168. (1) The Board may enter into an agreement...",
        "Effect to advance pricing agreement.\n169. \n[(1) Irrespective of anything to "
        "the contrary contained in section 263, where the modified return is a return "
        "furnished under section\n263.\n(3) Where a modified return is furnished...",
        "Secondary adjustment in certain cases.\n170. (1) An assessee shall make...",
    ])
    assert [c.number for c in chunks] == ["168", "169", "170"]


def test_section_numbers_stay_contiguous_across_the_real_act(chunks):
    """A leap between adjacent chunks means a stray match poisoned the fill."""
    sections = [c for c in chunks
                if c.doc_title.startswith("Income-tax Act") and c.unit == "Section"]
    leaps = [(b.ordinal, a.number, b.number)
             for a, b in zip(sections, sections[1:])
             if _sort_key(b.number)[0] - _sort_key(a.number)[0] > 5]
    assert not leaps, f"numbering leaps: {leaps[:5]}"
