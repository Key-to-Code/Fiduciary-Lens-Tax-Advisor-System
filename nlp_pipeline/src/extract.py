import re
import json
from shared.llm_client import get_provider

def extract_financial_entities(text: str) -> dict:
    """Extracts PAN, TAN, GSTIN, amounts, and complex entities from text."""
    entities = {
        "pan": [],
        "tan": [],
        "gstin": [],
        "amounts": [],
        "complex_entities": {}
    }
    
    if not text:
        return entities

    # Regex patterns for common Indian tax identifiers
    pan_pattern = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b')
    tan_pattern = re.compile(r'\b[A-Z]{4}[0-9]{5}[A-Z]{1}\b')
    gstin_pattern = re.compile(r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b')
    amount_pattern = re.compile(r'(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)', re.IGNORECASE)

    entities["pan"] = list(set(pan_pattern.findall(text)))
    entities["tan"] = list(set(tan_pattern.findall(text)))
    entities["gstin"] = list(set(gstin_pattern.findall(text)))
    entities["amounts"] = list(set(amount_pattern.findall(text)))

    # Use LLM for complex entities (Company Names, Deduction Sections)
    provider = get_provider()
    system_prompt = (
        "You are an expert entity extractor. Given the following text, extract any "
        "Company/Organization Names and Tax Deduction Sections claimed. "
        "Return ONLY a valid JSON object with keys 'organizations' (list of strings) "
        "and 'deductions' (list of strings). Do not return markdown, just raw JSON."
    )
    
    try:
        # Limit text size for entity extraction to avoid huge context costs
        llm_text = text[:10000]
        result = provider.complete(system=system_prompt, user=llm_text)
        
        # Strip markdown code blocks if the LLM adds them
        if result.startswith("```"):
            result = "\n".join(result.split("\n")[1:-1]).strip()
        if result.startswith("json"):
            result = result[4:].strip()
            
        complex_data = json.loads(result)
        entities["complex_entities"] = complex_data
    except Exception as e:
        entities["complex_entities"] = {"error": f"LLM extraction failed: {str(e)}"}

    return entities
