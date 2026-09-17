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
    
    # 1. Parse Document
    text = parse_document(file_path)
    
    # 2. Summarize
    print("Summarizing document...")
    summary = summarize_document(text)
    
    # 3. Extract Entities
    print("Extracting financial entities...")
    entities = extract_financial_entities(text)
    
    # 4. Tax Insights via RAG
    print("Generating tax insights via RAG...")
    rag_insights = {}
    
    deductions = entities.get("complex_entities", {}).get("deductions", [])
    qa_system = TaxQA()
    
    if deductions and isinstance(deductions, list):
        for deduction in deductions:
            question = f"What are the rules and limits for tax deduction under section {deduction}?"
            answer_obj = qa_system.ask(question)
            if answer_obj:
                rag_insights[deduction] = answer_obj.text
    else:
        # Fallback if no specific deductions found
        question = "What are the general tax saving investments available for a salaried employee in India?"
        answer_obj = qa_system.ask(question)
        if answer_obj:
            rag_insights["general"] = answer_obj.text
            
    return {
        "summary": summary,
        "entities": entities,
        "tax_insights": rag_insights
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = generate_insights(sys.argv[1])
        print("\n--- Pipeline Result ---")
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python insight.py <path_to_pdf_or_txt>")
