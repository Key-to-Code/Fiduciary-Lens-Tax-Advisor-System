import re
from shared.llm_client import get_provider

def summarize_document(text: str) -> str:
    """Summarizes a long tax document into a concise summary using the configured LLM."""
    if not text or not text.strip():
        return "No text provided to summarize."
        
    provider = get_provider()
    
    system_prompt = (
        "You are an expert Indian Chartered Accountant and tax advisor. "
        "Summarize the provided tax document or notice clearly and concisely. "
        "Highlight any action items, deadlines, and key financial figures."
    )
    
    # Simple chunking if the document is very long (naive approach for this example)
    max_chars = 15000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... [Text truncated for summarization] ..."

    return provider.complete(system=system_prompt, user=text)
