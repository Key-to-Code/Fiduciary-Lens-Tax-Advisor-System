import json
import sys
from pathlib import Path

# Add root to sys path so we can resolve nlp_pipeline, shared, and rag_model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Union
from nlp_pipeline.src.parse import parse_document
from nlp_pipeline.src.summarize import summarize_document
from nlp_pipeline.src.extract import extract_financial_entities
from rag_model.src.generation.answer import TaxQA

def generate_insights(file_path: Union[str, Path]) -> dict:
    """End-to-end pipeline to generate tax insights from a document."""
    print(f"Processing document: {file_path}")
    text = parse_document(file_path)
    return generate_insights_from_text(text)


def generate_insights_from_text(text: str) -> dict:
    """Run the existing summary, entity, and tax-RAG stages on parsed text."""
    if not text or not text.strip():
        raise ValueError("Document contains no text to analyze.")

    # 1. Summarize
    print("Summarizing document...")
    summary = summarize_document(text)

    # 3. Extract Entities
    print("Extracting financial entities...")
    entities = extract_financial_entities(text)
    
    # 4. Tax Insights via RAG
    print("Generating tax insights via RAG...")
    rag_insights = {}
    rag_sources = {}
    rag_grounding = {}

    def keep_answer(key, answer_obj):
        if not answer_obj:
            return
        rag_insights[key] = answer_obj.text
        rag_grounding[key] = bool(answer_obj.grounded)
        sources = []
        for source, hit in zip(answer_obj.sources, answer_obj.hits):
            item = dict(source)
            item["excerpt"] = hit.chunk.content[:900]
            item["url"] = hit.chunk.metadata.get("url")
            sources.append(item)
        rag_sources[key] = sources
    
    deductions = entities.get("complex_entities", {}).get("deductions", [])
    qa_system = TaxQA()
    
    if deductions and isinstance(deductions, list):
        for deduction in deductions:
            question = f"What are the rules and limits for tax deduction under section {deduction}?"
            answer_obj = qa_system.ask(question)
            keep_answer(deduction, answer_obj)
    else:
        # Fallback if no specific deductions found
        question = "What are the general tax saving investments available for a salaried employee in India?"
        answer_obj = qa_system.ask(question)
        keep_answer("general", answer_obj)
            
    return {
        "summary": summary,
        "entities": entities,
        "tax_insights": rag_insights,
        "tax_insight_sources": rag_sources,
        "tax_insight_grounding": rag_grounding,
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = generate_insights(sys.argv[1])
        print("\n--- Pipeline Result ---")
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python insight.py <path_to_pdf_or_txt>")
