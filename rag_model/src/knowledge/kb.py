from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from shared import config

_PROVISION_RE = re.compile(r"^(\d{1,3}[A-Z]{0,2})\.\s+(?=[\[(A-Z\"])", re.MULTILINE)
_MAX_PROVISION_JUMP = 10
_HEADING_RE = re.compile(r"^([A-Z][^\n]{2,110}\.)\s*$", re.MULTILINE)
_ANNEX_RE = re.compile(r"^(SCHEDULE\s+[IVXL]+|APPENDIX\s+[IVXL]+)\s*$", re.MULTILINE)
_ANNEX_TITLE_RE = re.compile(r"^[A-Z][A-Z ,\-()0-9./']{9,}$", re.MULTILINE)
_FA_SCHEDULE_RE = re.compile(r"^\s*THE\s+([A-Z]+)\s+SCHEDULE\s*$", re.MULTILINE)
_FA_PART_RE = re.compile(r"^\s*PART\s+([IVX]+)\s*$", re.MULTILINE)
_FA_PARAGRAPH_RE = re.compile(r"^\s*Paragraph\s+([A-E])\s*$", re.MULTILINE)
_FA_SUBPART_RE = re.compile(r"^\s*([AB])\.\s*[–—-]{1,3}\s*[^\n]*?INCOME-TAX ACT,\s*(\d{4})", re.MULTILINE | re.IGNORECASE)
_ORDINAL_WORDS = frozenset("First Second Third Fourth Fifth Sixth Seventh Eighth Ninth Tenth".split())
_STATUTE_ACT = {"title": "Income-tax Act, 2025", "unit": "Section", "kind": "statute", "note": "as amended by Finance Act, 2026; in force from 1 April 2026"}
_STATUTE_RULES = {"title": "Income-tax Rules, 2026", "unit": "Rule", "kind": "statute", "note": "G.S.R. 198(E) dated 20-3-2026; in force from 1 April 2026"}
_FINANCE_ACT = {"title": "Finance Act, 2026", "unit": "Section", "kind": "finance", "note": "No. 4 of 2026, assented 30-3-2026; rates for FY 2026-27"}
_DOC_PROFILES = ((re.compile(r"(?i)finance[-_ ]*act"), _FINANCE_ACT), (re.compile(r"(?i)income[-_ ]*tax[-_ ]*rules"), _STATUTE_RULES), (re.compile(r"(?i)income[-_ ]*tax[-_ ]*act"), _STATUTE_ACT), (re.compile(r"(?i)rules"), _STATUTE_RULES), (re.compile(r"(?i)act"), _STATUTE_ACT))

def _doc_profile(document_name: str) -> dict:
    """Returns the document profile based on the document name."""
    for pattern, profile in _DOC_PROFILES:
        if pattern.search(document_name):
            return profile
    return {"title": document_name, "unit": "Provision", "kind": "statute", "note": ""}

def _sort_key(number: str) -> tuple[int, str]:
    """Returns a tuple representing the sort key for a provision number."""
    digits = re.match(r"(\d+)([A-Z]*)", number)
    return (int(digits.group(1)), digits.group(2)) if digits else (0, "")

def _follows(candidate: str, current: str | None) -> bool:
    """Checks if a provision number can plausibly follow another."""
    if current is None:
        return True
    candidate_key, current_key = _sort_key(candidate), _sort_key(current)
    if candidate_key < current_key:
        return False
    return candidate_key[0] - current_key[0] <= _MAX_PROVISION_JUMP

@dataclass
class Chunk:
    """Represents a chunk of text with associated metadata."""
    chunk_id: str
    document_name: str
    content: str
    doc_title: str
    unit: str
    number: str | None
    heading: str | None
    source_note: str
    ordinal: int
    subdivision: str | None = None
    applies_to: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Returns a citation string for the chunk."""
        label = self.doc_title
        if self.number and self.number in _ORDINAL_WORDS:
            label += f", {self.number} {self.unit}"
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
        """Returns a short citation string for the chunk."""
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
    """Returns the marginal note immediately preceding a provision number."""
    window = text[max(0, position - 260):position]
    matches = list(_HEADING_RE.finditer(window))
    if not matches:
        return None
    if window[matches[-1].end():].strip():
        return None
    candidate = matches[-1].group(1).strip()
    return candidate if not candidate[0].isdigit() else None

def _annex_title(text: str, position: int) -> str | None:
    """Returns the ALL-CAPS descriptive title under a SCHEDULE/APPENDIX banner."""
    match = _ANNEX_TITLE_RE.search(text, position)
    return match.group(0).strip().title() if match else None

def load_chunks(kb_path: Path | None = None) -> list[Chunk]:
    """Loads the knowledge base and returns chunks with resolved provision metadata."""
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
    """Loads a statute chunk and returns it with resolved provision metadata."""
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
        return _build(record, ordinal, profile, unit, number, heading)

    opens_at = _PROVISION_RE.search(text)
    if opens_at:
        candidate = opens_at.group(1)
        if _follows(candidate, number) and (opens_at.start() < 400 or number is None):
            number = candidate
            heading = _heading_before(text, opens_at.start())

    built = _build(record, ordinal, profile, unit, number, heading)

    for match in reversed(list(_PROVISION_RE.finditer(text))):
        candidate = match.group(1)
        if _follows(candidate, number):
            state[document] = (unit, candidate, _heading_before(text, match.start()))
            break
    else:
        state[document] = (unit, number, heading)

    return built

def _finance_markers(text: str) -> list[tuple[int, str, str]]:
    """Returns every structural boundary in a Finance Act chunk."""
    found = [(m.start(), "schedule", m.group(1).title()) for m in _FA_SCHEDULE_RE.finditer(text)]
    found += [(m.start(), "part", m.group(1)) for m in _FA_PART_RE.finditer(text)]
    found += [(m.start(), "paragraph", m.group(1)) for m in _FA_PARAGRAPH_RE.finditer(text)]
    found += [(m.start(), "subpart", m.group(2)) for m in _FA_SUBPART_RE.finditer(text)]
    found += [(m.start(), "section", m.group(1)) for m in _PROVISION_RE.finditer(text)]
    return sorted(found)

def _apply_finance_marker(state: dict, kind: str, value: str) -> None:
    """Applies a finance marker to the state."""
    if kind == "schedule":
        state.update(schedule=value, part=None, paragraph=None, applies_to=None)
    elif kind == "part":
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
    """Loads a finance chunk and returns it with resolved provision metadata."""
    document = record["document_name"]
    text = record["content"]
    entering = dict(state.get(document, {}))

    markers = _finance_markers(text)
    labelled = dict(entering)
    for position, kind, value in markers:
        if position < 400:
            _apply_finance_marker(labelled, kind, value)

    exiting = dict(entering)
    for _, kind, value in markers:
        _apply_finance_marker(exiting, kind, value)
    state[document] = exiting

    spanned = {value for _, kind, value in markers if kind == "subpart"}
    reset_here = any(kind in ("schedule", "part") for _, kind, _ in markers)
    if entering.get("applies_to") and not reset_here:
        spanned.add(entering["applies_to"].rsplit(" ", 1)[-1])
    if len(spanned) > 1:
        years = ", ".join(f"Income-tax Act, {year}" for year in sorted(spanned))
        labelled["applies_to"] = f"{years} - passage spans both, check the text"

    if labelled.get("schedule"):
        subdivision = ", ".join(part for part in (f"Part {labelled['part']}" if labelled.get("part") else None, f"Paragraph {labelled['paragraph']}" if labelled.get("paragraph") else None) if part) or None
        return _build(record, ordinal, profile, "Schedule", labelled["schedule"], None, subdivision=subdivision, applies_to=labelled.get("applies_to"))

    return _build(record, ordinal, profile, "Section", labelled.get("section"), None, applies_to=labelled.get("applies_to"))

def _build(record, ordinal, profile, unit, number, heading, subdivision=None, applies_to=None) -> Chunk:
    """Builds a chunk with the given metadata."""
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

_PART_ALIASES = {
    "I": "rates of income-tax, tax slabs, slab rates, tax brackets, rate of tax on total income for individuals, companies and firms",
    "II": "rates for deduction of tax at source, TDS rates",
    "III": "rates of income-tax, tax slabs, slab rates, tax brackets, rate of tax on total income, deduction of tax from salaries, rates for computing advance tax",
    "IV": "rules for computation of net agricultural income",
}

def index_text(chunk: Chunk) -> str:
    """Returns the text to be indexed, prefixed with the provision label."""
    header = chunk.citation
    if chunk.unit == "Schedule" and chunk.number == "First" and chunk.subdivision:
        part = chunk.subdivision.split(",")[0].removeprefix("Part ").strip()
        alias = _PART_ALIASES.get(part)
        if alias:
            header += f"\n{alias}"
    return f"{header}\n{chunk.content}"