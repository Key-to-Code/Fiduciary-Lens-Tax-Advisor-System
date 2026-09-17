import pymupdf
from pathlib import Path

def parse_document(file_path: str | Path) -> str:
    """Parses a document (PDF or text) and returns its content as a string."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    if path.suffix.lower() == '.pdf':
        return parse_pdf(path)
    elif path.suffix.lower() in ['.txt', '.md', '.csv']:
        return path.read_text(encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Supported formats are .pdf, .txt, .md, .csv")

def parse_pdf(file_path: Path) -> str:
    """Extracts text from a PDF file using PyMuPDF."""
    text_content = []
    try:
        with pymupdf.open(file_path) as doc:
            for page in doc:
                text_content.append(page.get_text())
        return "\n".join(text_content)
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF {file_path}: {e}")
