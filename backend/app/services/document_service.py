"""
Document upload service.

Handles all upload validation, secure temp-file management, and text extraction.
Text extraction is delegated exclusively to the EXISTING nlp_pipeline/src/parse.py
— no new parsing logic is introduced here.

Security model
--------------
* The client-supplied filename is used for **display only**.  The actual file on
  disk is written to an OS-managed temp path produced by tempfile.mkstemp().
  Path traversal is structurally impossible.
* The temp file is always deleted in a `finally` block regardless of success or
  failure, so no document data lingers on disk.
* No filesystem paths are returned to the caller.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from backend.app.core.config import settings

# ---------------------------------------------------------------------------
# Allowed formats — mirrors nlp_pipeline/src/parse.py exactly
# ---------------------------------------------------------------------------
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md", ".csv"})

# Mapping from extension to the canonical MIME type we report back.
# We deliberately accept any Content-Type the client sends (browsers often
# set it wrong) and re-derive the MIME type from the validated extension.
_EXT_TO_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md":  "text/markdown",
    ".csv": "text/csv",
}

# PDF magic bytes — first 4 bytes of any valid PDF are b"%PDF"
_PDF_MAGIC = b"%PDF"

_PREVIEW_CHARS = 500


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def process_upload(upload: UploadFile) -> dict[str, Any]:
    """
    Validate and extract text from an uploaded document.

    Parameters
    ----------
    upload:
        The FastAPI UploadFile from the multipart request.

    Returns
    -------
    dict matching DocumentUploadResponse schema fields.

    Raises
    ------
    ValueError
        For client-side validation failures (bad type, empty file, etc.).
    RuntimeError
        When the file is too large (sentinel for 413).
    OSError
        Propagated from parse_document when the underlying parser fails.
    """
    # ── 1. Derive and validate the file extension ────────────────────────────
    original_name = upload.filename or "upload"
    # Strip any directory components the client might include (defence-in-depth).
    safe_display_name = Path(original_name).name or "upload"
    extension = Path(safe_display_name).suffix.lower()

    if extension not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type",
            f"'{extension or '(none)'}' is not supported. Allowed: {allowed}",
        )

    # ── 2. Read raw bytes ────────────────────────────────────────────────────
    raw_bytes: bytes = await upload.read()

    # ── 3. Empty-file check ──────────────────────────────────────────────────
    if len(raw_bytes) == 0:
        raise ValueError("Empty file", "The uploaded file contains no data.")

    # ── 4. Size limit ────────────────────────────────────────────────────────
    max_bytes = settings.UPLOAD_MAX_SIZE_MB * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise RuntimeError(
            f"File too large",
            f"Upload is {len(raw_bytes):,} bytes; limit is "
            f"{settings.UPLOAD_MAX_SIZE_MB} MB ({max_bytes:,} bytes).",
        )

    # ── 5. PDF magic-bytes sanity check ─────────────────────────────────────
    if extension == ".pdf" and not raw_bytes[:4].startswith(_PDF_MAGIC):
        raise ValueError(
            "Invalid PDF",
            "The file does not appear to be a valid PDF (missing %PDF header).",
        )

    # ── 6. Write to a secure OS-managed temp file ────────────────────────────
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=extension)
    tmp_path = Path(tmp_path_str)

    try:
        # Write bytes through the file descriptor then close it so parse_document
        # can open the file on any OS (Windows locks open fds).
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(raw_bytes)

        # ── 7. Extract text via the EXISTING parser ──────────────────────────
        # Deferred import — keeps server startup instant; pymupdf loads only
        # when needed. nlp_pipeline/src/parse.py imports pymupdf at module
        # level, so if the root requirements.txt hasn't been pip-installed yet,
        # importing the module raises ImportError for ALL file types (not just
        # PDF). We catch that and surface a clear dependency message for PDFs,
        # while falling back to a direct read for plain-text formats (which is
        # exactly what parse.py does internally for those types).
        try:
            from nlp_pipeline.src.parse import parse_document  # noqa: PLC0415
            extracted_text: str = parse_document(tmp_path)
        except ImportError as exc:
            if extension == ".pdf":
                raise RuntimeError(
                    "PDF parser unavailable",
                    "PyMuPDF is not installed. Run: pip install -r requirements.txt",
                ) from exc
            # For .txt / .md / .csv the parse.py logic is just read_text —
            # replicate that single line so text files work without pymupdf.
            extracted_text = tmp_path.read_text(encoding="utf-8")

    finally:
        # Always remove the temp file — whether extraction succeeded or not.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass  # best-effort cleanup; not worth propagating

    # ── 8. Build result ──────────────────────────────────────────────────────
    doc_id = str(uuid.uuid4())
    char_count = len(extracted_text)
    preview = extracted_text[:_PREVIEW_CHARS].strip()

    return {
        "success": True,
        "document_id": doc_id,
        "filename": safe_display_name,   # display only, never a path
        "file_size_bytes": len(raw_bytes),
        "mime_type": _EXT_TO_MIME[extension],
        "char_count": char_count,
        "preview": preview,
        "message": "Document uploaded and text extracted successfully",
    }
