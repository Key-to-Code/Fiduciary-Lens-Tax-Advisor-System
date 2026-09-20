"""
Document upload and text extraction service.

Handles upload validation, secure temp-file lifecycle, and text extraction.
Text extraction is delegated exclusively to the EXISTING nlp_pipeline/src/parse.py
— no second parsing implementation is introduced.

Security model
--------------
* Client-supplied filenames are sanitized with Path(...).name and used for
  display/logging only; never used in internal filesystem paths.
* Temp files are created with tempfile.mkstemp() in the OS temp directory.
* Temp files are ALWAYS deleted in a `finally` block regardless of outcome.
* No internal paths or stack traces are leaked to the client.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from backend.app.core.config import settings

# ---------------------------------------------------------------------------
# Allowed formats — strictly mirrors nlp_pipeline/src/parse.py
# ---------------------------------------------------------------------------
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md", ".csv"})

_EXT_TO_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md":  "text/markdown",
    ".csv": "text/csv",
}

_PDF_MAGIC = b"%PDF"
_PREVIEW_CHARS = 500
_MIN_TEXT_CHARS = 5
_MAX_FILENAME_CHARS = 255


@dataclass
class ExtractedDocument:
    """Represents a validated and parsed document."""

    document_id: str
    filename: str
    file_size_bytes: int
    mime_type: str
    text: str
    char_count: int


def sanitize_display_filename(filename: str | None, *, default: str = "document.txt") -> str:
    """Return a database-safe display name without accepting a client path.

    UploadFile names may originate from Windows clients, where a backslash is a
    path separator even on this POSIX host. The name is never used as a storage
    path, but normalising both separators avoids returning a client path and
    keeps direct-text summaries consistent with file uploads.
    """
    candidate = (filename or "").replace("\\", "/").replace("\x00", "")
    candidate = Path(candidate).name.strip()
    if not candidate or candidate in {".", ".."}:
        candidate = default

    if len(candidate) <= _MAX_FILENAME_CHARS:
        return candidate

    suffix = Path(candidate).suffix
    if len(suffix) >= _MAX_FILENAME_CHARS:
        return candidate[-_MAX_FILENAME_CHARS:]
    stem_limit = _MAX_FILENAME_CHARS - len(suffix)
    return f"{Path(candidate).stem[:stem_limit]}{suffix}"


async def extract_document_text(upload: UploadFile) -> ExtractedDocument:
    """
    Validate an uploaded file, write it to a secure temp path, extract its text
    using the existing NLP pipeline parser, validate the extracted text, and clean
    up the temporary file.

    Parameters
    ----------
    upload:
        The FastAPI UploadFile from the request.

    Returns
    -------
    ExtractedDocument
        Data object with metadata and extracted text.

    Raises
    ------
    ValueError
        For validation errors (unsupported type, empty file, invalid PDF header,
        corrupted document, empty or insufficient extracted text).
    RuntimeError
        When the file size exceeds the configured maximum limit.
    """
    # ── 1. Validate file extension ──────────────────────────────────────────
    safe_display_name = sanitize_display_filename(upload.filename)
    extension = Path(safe_display_name).suffix.lower()

    if extension not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        raise ValueError(
            "Unsupported file type",
            f"'{extension or '(none)'}' is not supported. Allowed: {allowed}",
        )

    # ── 2. Read raw bytes ────────────────────────────────────────────────────
    raw_bytes: bytes = await upload.read()

    # ── 3. Empty file check ──────────────────────────────────────────────────
    if len(raw_bytes) == 0:
        raise ValueError("Empty file", "The uploaded file contains no data.")

    # ── 4. Size limit check ──────────────────────────────────────────────────
    max_bytes = settings.UPLOAD_MAX_SIZE_MB * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise RuntimeError(
            "File too large",
            f"Upload is {len(raw_bytes):,} bytes; limit is "
            f"{settings.UPLOAD_MAX_SIZE_MB} MB ({max_bytes:,} bytes).",
        )

    # ── 5. PDF magic header check ────────────────────────────────────────────
    if extension == ".pdf" and not raw_bytes[:4].startswith(_PDF_MAGIC):
        raise ValueError(
            "Invalid PDF",
            "The file does not appear to be a valid PDF (missing %PDF header).",
        )

    # ── 6. Write to secure OS-managed temp file ──────────────────────────────
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=extension)
    tmp_path = Path(tmp_path_str)

    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(raw_bytes)

        # ── 7. Extract text via the EXISTING parser ──────────────────────────
        try:
            from nlp_pipeline.src.parse import parse_document  # noqa: PLC0415
            extracted_text: str = parse_document(tmp_path)
        except ImportError as exc:
            if extension == ".pdf":
                raise RuntimeError(
                    "PDF parser unavailable",
                    "PyMuPDF is not installed. Run: pip install -r requirements.txt",
                ) from exc
            extracted_text = tmp_path.read_text(encoding="utf-8")
        except Exception as exc:
            # Trap parser exceptions (e.g. PyMuPDF corrupted file errors)
            raise ValueError(
                "Corrupted document",
                "Failed to parse document: file appears corrupted or unreadable.",
            ) from exc

        # ── 8. Validate extracted content ────────────────────────────────────
        if not extracted_text or not extracted_text.strip():
            raise ValueError(
                "Empty document content",
                "The document contains no extractable text.",
            )

        if len(extracted_text.strip()) < _MIN_TEXT_CHARS:
            raise ValueError(
                "Insufficient document content",
                "The document does not contain sufficient text for legal processing.",
            )

    finally:
        # Guarantee cleanup of temporary file
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    doc_id = str(uuid.uuid4())
    return ExtractedDocument(
        document_id=doc_id,
        filename=safe_display_name,
        file_size_bytes=len(raw_bytes),
        mime_type=_EXT_TO_MIME[extension],
        text=extracted_text,
        char_count=len(extracted_text),
    )


async def process_upload(upload: UploadFile) -> dict[str, Any]:
    """
    Backward-compatible upload processor for Stage 2 /api/v1/documents/upload.
    """
    doc = await extract_document_text(upload)
    preview = doc.text[:_PREVIEW_CHARS].strip()

    return {
        "success": True,
        "document_id": doc.document_id,
        "filename": doc.filename,
        "file_size_bytes": doc.file_size_bytes,
        "mime_type": doc.mime_type,
        "char_count": doc.char_count,
        "preview": preview,
        "message": "Document uploaded and text extracted successfully",
    }
