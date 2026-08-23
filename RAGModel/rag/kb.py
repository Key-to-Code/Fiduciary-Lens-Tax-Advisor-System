"""Load the chunked knowledge base and attach trustworthy citation metadata.

The chunker that produced `rag_knowledge_base.json` stored a `metadata.section`
guessed with a loose regex, which fires on any mention of a section ("sub-section"
-> "TION"; the cross-reference inside the definition of "accountant" -> 515).
A citation has to name the provision a passage *is*, not one it mentions, so we
re-derive it here by walking each document in reading order and carrying the
current provision forward across chunk boundaries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

# A provision opens with its number at the start of a line, followed on the SAME
# line by its text: "2. (1) In these rules..." / "228. (1) An undertaking...".
# Requiring same-line text rejects schedule-table row numbers, which sit alone.
_PROVISION_RE = re.compile(r"^(\d{1,3}[A-Z]{0,2})\.\s+(?=[(A-Z\"])", re.MULTILINE)

# Marginal note / heading: a short line ending in a period, e.g. "Definitions."
_HEADING_RE = re.compile(r"^([A-Z][^\n]{2,110}\.)\s*$", re.MULTILINE)

# The Act's schedules and the Rules' appendices restart their own numbering, so
# they end the running section count rather than continuing it.
_ANNEX_RE = re.compile(r"^(SCHEDULE\s+[IVXL]+|APPENDIX\s+[IVXL]+)\s*$", re.MULTILINE)
_ANNEX_TITLE_RE = re.compile(r"^[A-Z][A-Z ,\-()0-9./']{9,}$", re.MULTILINE)

_DOCS = {
    "Act": {"title": "Income-tax Act, 2025", "unit": "Section",
            "note": "as amended by Finance Act, 2026; in force from 1 April 2026"},
    "Rules": {"title": "Income-tax Rules, 2026", "unit": "Rule",
              "note": "G.S.R. 198(E) dated 20-3-2026; in force from 1 April 2026"},
}


def _doc_profile(document_name: str) -> dict:
    for key, profile in _DOCS.items():
        if key.lower() in document_name.lower():
            return profile
    return {"title": document_name, "unit": "Provision", "note": ""}


def _sort_key(number: str) -> tuple[int, str]:
    digits = re.match(r"(\d+)([A-Z]*)", number)
    return (int(digits.group(1)), digits.group(2)) if digits else (0, "")


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
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """e.g. 'Income-tax Act, 2025, Section 17 (Salary)'."""
        label = self.doc_title
        if self.number:
            label += f", {self.unit} {self.number}"
        if self.heading:
            label += f" ({self.heading.rstrip('.')})"
        return label

    @property
    def short_citation(self) -> str:
        """Compact form for inline use, e.g. 'Act s.17' / 'Rules r.228'."""
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
    # document -> (unit, number, heading) currently in force at the chunk boundary
    state: dict[str, tuple[str, str | None, str | None]] = {}

    for ordinal, record in enumerate(raw):
        document = record["document_name"]
        text = record["content"]
        profile = _doc_profile(document)
        base_unit = profile["unit"]

        unit, number, heading = state.get(document, (base_unit, None, None))

        annex = _ANNEX_RE.search(text)
        if annex:
            kind, roman = annex.group(1).split()
            unit, number = kind.title(), roman
            heading = _annex_title(text, annex.end())
            state[document] = (unit, number, heading)
            chunks.append(_build(record, ordinal, profile, unit, number, heading))
            continue

        if unit in ("Schedule", "Appendix"):
            # Inside an annex the numbering is local; keep the annex as the citation.
            chunks.append(_build(record, ordinal, profile, unit, number, heading))
            continue

        opens_at = _PROVISION_RE.search(text)
        if opens_at:
            candidate = opens_at.group(1)
            # Statutes run in ascending order, so a lower number is a false
            # positive (a quoted list, a table row) rather than a new provision.
            ascending = number is None or _sort_key(candidate) >= _sort_key(number)
            # Attribute the chunk to the new provision only when the chunk
            # essentially starts there; otherwise most of it is still the old one.
            if ascending and (opens_at.start() < 400 or number is None):
                number = candidate
                heading = _heading_before(text, opens_at.start())

        chunks.append(_build(record, ordinal, profile, unit, number, heading))

        # Whatever provision the chunk *ends* in carries into the next chunk.
        for match in reversed(list(_PROVISION_RE.finditer(text))):
            candidate = match.group(1)
            if number is None or _sort_key(candidate) >= _sort_key(number):
                state[document] = (unit, candidate, _heading_before(text, match.start()))
                break
        else:
            state[document] = (unit, number, heading)

    return chunks


def _build(record, ordinal, profile, unit, number, heading) -> Chunk:
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
        metadata=record.get("metadata", {}),
    )


def index_text(chunk: Chunk) -> str:
    """What actually gets embedded: the passage prefixed by its provision label.

    Folding the citation into the embedded text lets a query that names a
    provision ("what does section 17 cover") match the label as well as the prose.
    """
    return f"{chunk.citation}\n{chunk.content}"
