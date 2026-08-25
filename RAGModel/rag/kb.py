"""Load the chunked knowledge base and attach trustworthy citation metadata.

The chunker that produced `rag_knowledge_base.json` stored a `metadata.section`
guessed with a loose regex, which fires on any *mention* of a section
("sub-section" -> "TION"; the cross-reference inside the definition of
"accountant" -> 515). A citation has to name the provision a passage *is*, not
one it mentions, so we re-derive it here by walking each document in reading
order and carrying the current provision across chunk boundaries.

Two document shapes are handled:

*statute* (Income-tax Act, Income-tax Rules)
    A flat ascending run of numbered provisions, then Schedules/Appendices that
    restart their own numbering.

*finance* (the annual Finance Act)
    A body of numbered sections, then Schedules organised as
    Schedule -> Part -> Sub-part -> Paragraph. The sub-part matters more than
    anything else here: the First Schedule carries rates for BOTH the old
    Income-tax Act, 1961 (sub-part A) and the Income-tax Act, 2025 (sub-part B).
    Citing a 1961 slab table for a 2025 Act question would be confidently wrong,
    so the governing Act is resolved per chunk and carried into the citation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

# A provision opens with its number at the start of a line, followed by its text:
# "2. (1) In these rules..." / "228. (1) An undertaking...". The bracket is in the
# lookahead because amended provisions render as "169.\n[(1) Irrespective..." --
# omitting it silently skipped every substituted section in the Act.
_PROVISION_RE = re.compile(r"^(\d{1,3}[A-Z]{0,2})\.\s+(?=[\[(A-Z\"])", re.MULTILINE)

# A cross-reference that wraps ("...furnished under section\n263.") is
# indistinguishable from a provision opening by line position alone. In a linear
# statute consecutive provisions advance by a step or two, so a large forward
# leap is a cross-reference rather than the next section. Without this guard one
# stray match poisons the forward-fill: a jump to 263 at section 168 made every
# genuine section from 169 to 262 inherit the wrong citation.
_MAX_PROVISION_JUMP = 10

# Marginal note / heading: a short line ending in a period, e.g. "Definitions."
_HEADING_RE = re.compile(r"^([A-Z][^\n]{2,110}\.)\s*$", re.MULTILINE)

# The Act's schedules and the Rules' appendices restart their own numbering, so
# they end the running section count rather than continuing it.
_ANNEX_RE = re.compile(r"^(SCHEDULE\s+[IVXL]+|APPENDIX\s+[IVXL]+)\s*$", re.MULTILINE)
_ANNEX_TITLE_RE = re.compile(r"^[A-Z][A-Z ,\-()0-9./']{9,}$", re.MULTILINE)

# --- Finance Act structure ----------------------------------------------
_FA_SCHEDULE_RE = re.compile(r"^\s*THE\s+([A-Z]+)\s+SCHEDULE\s*$", re.MULTILINE)
_FA_PART_RE = re.compile(r"^\s*PART\s+([IVX]+)\s*$", re.MULTILINE)
_FA_PARAGRAPH_RE = re.compile(r"^\s*Paragraph\s+([A-E])\s*$", re.MULTILINE)
# "A.--INCOME-TAX UNDER THE INCOME-TAX ACT, 1961" / "B.--UNDER THE ... ACT, 2025"
_FA_SUBPART_RE = re.compile(
    r"^\s*([AB])\.\s*[–—-]{1,3}\s*[^\n]*?INCOME-TAX ACT,\s*(\d{4})",
    re.MULTILINE | re.IGNORECASE,
)

_ORDINAL_WORDS = frozenset(
    "First Second Third Fourth Fifth Sixth Seventh Eighth Ninth Tenth".split()
)

_STATUTE_ACT = {
    "title": "Income-tax Act, 2025", "unit": "Section", "kind": "statute",
    "note": "as amended by Finance Act, 2026; in force from 1 April 2026",
}
_STATUTE_RULES = {
    "title": "Income-tax Rules, 2026", "unit": "Rule", "kind": "statute",
    "note": "G.S.R. 198(E) dated 20-3-2026; in force from 1 April 2026",
}
_FINANCE_ACT = {
    "title": "Finance Act, 2026", "unit": "Section", "kind": "finance",
    "note": "No. 4 of 2026, assented 30-3-2026; rates for FY 2026-27",
}

# Ordered, most specific first: a Finance Act filename also contains "act", so
# matching must not fall through to the Income-tax Act profile.
_DOC_PROFILES: tuple[tuple[re.Pattern, dict], ...] = (
    (re.compile(r"(?i)finance[-_ ]*act"), _FINANCE_ACT),
    (re.compile(r"(?i)income[-_ ]*tax[-_ ]*rules"), _STATUTE_RULES),
    (re.compile(r"(?i)income[-_ ]*tax[-_ ]*act"), _STATUTE_ACT),
    (re.compile(r"(?i)rules"), _STATUTE_RULES),
    (re.compile(r"(?i)act"), _STATUTE_ACT),
)


def _doc_profile(document_name: str) -> dict:
    for pattern, profile in _DOC_PROFILES:
        if pattern.search(document_name):
            return profile
    return {"title": document_name, "unit": "Provision", "kind": "statute", "note": ""}


def _sort_key(number: str) -> tuple[int, str]:
    digits = re.match(r"(\d+)([A-Z]*)", number)
    return (int(digits.group(1)), digits.group(2)) if digits else (0, "")


def _follows(candidate: str, current: str | None) -> bool:
    """Whether `candidate` can plausibly be the provision after `current`."""
    if current is None:
        return True
    candidate_key, current_key = _sort_key(candidate), _sort_key(current)
    if candidate_key < current_key:
        return False    # statutes do not run backwards
    return candidate_key[0] - current_key[0] <= _MAX_PROVISION_JUMP


@dataclass
class Chunk:
    chunk_id: str
    document_name: str
    content: str
    doc_title: str
    unit: str            # "Section", "Rule", "Schedule" or "Appendix"
    number: str | None   # provision this passage belongs to, e.g. "17" or "XV"
    heading: str | None  # marginal note, e.g. "Definitions."
    source_note: str
    ordinal: int
    subdivision: str | None = None   # e.g. "Part III, Paragraph A"
    applies_to: str | None = None    # Act whose rates a Finance Act passage sets
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """e.g. 'Income-tax Act, 2025, Section 17 (Salary)'."""
        label = self.doc_title
        if self.number and self.number in _ORDINAL_WORDS:
            label += f", {self.number} {self.unit}"   # "First Schedule"
        elif self.number:
            label += f", {self.unit} {self.number}"
        if self.subdivision:
            label += f", {self.subdivision}"
        if self.heading:
            label += f" ({self.heading.rstrip('.')})"
        if self.applies_to:
            label += f" [rates under the {self.applies_to}]"
        return label

    @property
    def short_citation(self) -> str:
        """Compact form for inline use, e.g. 'Act s.17' / 'Rules r.228'."""
        if self.doc_title.startswith("Finance Act"):
            if self.number in _ORDINAL_WORDS:
                schedule = f"FA 2026 {self.number} Sch"
                return f"{schedule}, {self.subdivision}" if self.subdivision else schedule
            return f"FA 2026 s.{self.number}"
        book = "Act" if self.doc_title.startswith("Income-tax Act") else "Rules"
        if not self.number:
            return book
        marker = {"Section": "s.", "Rule": "r."}.get(self.unit, f"{self.unit} ")
        return f"{book} {marker}{self.number}"


def _heading_before(text: str, position: int) -> str | None:
    """The marginal note immediately preceding a provision number, if any."""
    window = text[max(0, position - 260):position]
    matches = list(_HEADING_RE.finditer(window))
    if not matches:
        return None
    # A sentence tail can look like a heading; require it to sit right on the number.
    if window[matches[-1].end():].strip():
        return None
    candidate = matches[-1].group(1).strip()
    return candidate if not candidate[0].isdigit() else None


def _annex_title(text: str, position: int) -> str | None:
    """The ALL-CAPS descriptive title under a SCHEDULE/APPENDIX banner."""
    match = _ANNEX_TITLE_RE.search(text, position)
    return match.group(0).strip().title() if match else None


def load_chunks(kb_path: Path | None = None) -> list[Chunk]:
    """Read the KB JSON and return chunks carrying resolved provision metadata."""
    kb_path = Path(kb_path or config.KB_JSON)
    raw = json.loads(kb_path.read_text(encoding="utf-8"))

    chunks: list[Chunk] = []
    statute_state: dict[str, tuple[str, str | None, str | None]] = {}
    finance_state: dict[str, dict] = {}

    for ordinal, record in enumerate(raw):
        profile = _doc_profile(record["document_name"])
        if profile["kind"] == "finance":
            chunks.append(_load_finance_chunk(record, ordinal, profile, finance_state))
        else:
            chunks.append(_load_statute_chunk(record, ordinal, profile, statute_state))

    return chunks


def _load_statute_chunk(record, ordinal, profile, state) -> Chunk:
    document = record["document_name"]
    text = record["content"]
    unit, number, heading = state.get(document, (profile["unit"], None, None))

    annex = _ANNEX_RE.search(text)
    if annex:
        kind, roman = annex.group(1).split()
        unit, number = kind.title(), roman
        heading = _annex_title(text, annex.end())
        state[document] = (unit, number, heading)
        return _build(record, ordinal, profile, unit, number, heading)

    if unit in ("Schedule", "Appendix"):
        # Inside an annex the numbering is local; keep the annex as the citation.
        return _build(record, ordinal, profile, unit, number, heading)

    opens_at = _PROVISION_RE.search(text)
    if opens_at:
        candidate = opens_at.group(1)
        # Attribute the chunk to the new provision only when the chunk essentially
        # starts there; otherwise most of it is still the old one.
        if _follows(candidate, number) and (opens_at.start() < 400 or number is None):
            number = candidate
            heading = _heading_before(text, opens_at.start())

    built = _build(record, ordinal, profile, unit, number, heading)

    # Whatever provision the chunk *ends* in carries into the next chunk.
    for match in reversed(list(_PROVISION_RE.finditer(text))):
        candidate = match.group(1)
        if _follows(candidate, number):
            state[document] = (unit, candidate, _heading_before(text, match.start()))
            break
    else:
        state[document] = (unit, number, heading)

    return built


def _finance_markers(text: str) -> list[tuple[int, str, str]]:
    """Every structural boundary in a Finance Act chunk, in reading order."""
    found = [(m.start(), "schedule", m.group(1).title()) for m in _FA_SCHEDULE_RE.finditer(text)]
    found += [(m.start(), "part", m.group(1)) for m in _FA_PART_RE.finditer(text)]
    found += [(m.start(), "paragraph", m.group(1)) for m in _FA_PARAGRAPH_RE.finditer(text)]
    found += [(m.start(), "subpart", m.group(2)) for m in _FA_SUBPART_RE.finditer(text)]
    found += [(m.start(), "section", m.group(1)) for m in _PROVISION_RE.finditer(text)]
    return sorted(found)


def _apply_finance_marker(state: dict, kind: str, value: str) -> None:
    if kind == "schedule":
        state.update(schedule=value, part=None, paragraph=None, applies_to=None)
    elif kind == "part":
        # Part numbers appear in the table of contents too, before any schedule
        # banner; only treat them as structure once a schedule has opened.
        if state.get("schedule"):
            state.update(part=value, paragraph=None)
    elif kind == "paragraph":
        state["paragraph"] = value
    elif kind == "subpart":
        state["applies_to"] = f"Income-tax Act, {value}"
    elif kind == "section":
        if not state.get("schedule"):
            state["section"] = value


def _load_finance_chunk(record, ordinal, profile, state) -> Chunk:
    document = record["document_name"]
    text = record["content"]
    entering = dict(state.get(document, {}))

    markers = _finance_markers(text)
    # Same rule as the statute path: the chunk is labelled by the structure in
    # force near its start, while every marker carries forward to the next chunk.
    labelled = dict(entering)
    for position, kind, value in markers:
        if position < 400:
            _apply_finance_marker(labelled, kind, value)

    exiting = dict(entering)
    for _, kind, value in markers:
        _apply_finance_marker(exiting, kind, value)
    state[document] = exiting

    # A chunk can straddle the A/B sub-part boundary, where the governing Act
    # changes from 1961 to 2025 mid-passage. Labelling such a chunk with either
    # one alone is the exact failure this metadata exists to prevent, so say that
    # it spans both and let the reader see it rather than trusting one label.
    spanned = {value for _, kind, value in markers if kind == "subpart"}
    # A Schedule or Part banner already reset the governing Act, so what was in
    # force before it is not part of what this chunk spans.
    reset_here = any(kind in ("schedule", "part") for _, kind, _ in markers)
    if entering.get("applies_to") and not reset_here:
        spanned.add(entering["applies_to"].rsplit(" ", 1)[-1])
    if len(spanned) > 1:
        years = ", ".join(f"Income-tax Act, {year}" for year in sorted(spanned))
        labelled["applies_to"] = f"{years} - passage spans both, check the text"

    if labelled.get("schedule"):
        subdivision = ", ".join(
            part for part in (
                f"Part {labelled['part']}" if labelled.get("part") else None,
                f"Paragraph {labelled['paragraph']}" if labelled.get("paragraph") else None,
            ) if part
        ) or None
        return _build(record, ordinal, profile, "Schedule", labelled["schedule"],
                      None, subdivision=subdivision,
                      applies_to=labelled.get("applies_to"))

    return _build(record, ordinal, profile, "Section", labelled.get("section"),
                  None, applies_to=labelled.get("applies_to"))


def _build(record, ordinal, profile, unit, number, heading,
           subdivision=None, applies_to=None) -> Chunk:
    return Chunk(
        chunk_id=record["chunk_id"],
        document_name=record["document_name"],
        content=record["content"],
        doc_title=profile["title"],
        unit=unit,
        number=number,
        heading=heading,
        source_note=profile["note"],
        ordinal=ordinal,
        subdivision=subdivision,
        applies_to=applies_to,
        metadata=record.get("metadata", {}),
    )


# Vocabulary bridge for the Finance Act's First Schedule. The statute says "rates
# of income-tax"; users say "slab", "bracket", "how much tax". The rate tables are
# also mostly numerals, so they carry almost no semantic signal of their own and
# lose to any section that merely mentions "individual". These aliases restate
# what each Part's own heading says -- they add retrieval vocabulary, never law --
# and are folded into the embedded text only, never into what the model reads.
_PART_ALIASES = {
    "I": "rates of income-tax, tax slabs, slab rates, tax brackets, "
         "rate of tax on total income for individuals, companies and firms",
    "II": "rates for deduction of tax at source, TDS rates",
    "III": "rates of income-tax, tax slabs, slab rates, tax brackets, "
           "rate of tax on total income, deduction of tax from salaries, "
           "rates for computing advance tax",
    "IV": "rules for computation of net agricultural income",
}


def index_text(chunk: Chunk) -> str:
    """What actually gets embedded: the passage prefixed by its provision label.

    Folding the citation into the embedded text lets a query that names a
    provision ("what does section 17 cover") match the label as well as the prose.
    """
    header = chunk.citation
    if chunk.unit == "Schedule" and chunk.number == "First" and chunk.subdivision:
        part = chunk.subdivision.split(",")[0].removeprefix("Part ").strip()
        alias = _PART_ALIASES.get(part)
        if alias:
            header += f"\n{alias}"
    return f"{header}\n{chunk.content}"
