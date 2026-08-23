import os
import json
import re
import fitz  # PyMuPDF for superior document parsing
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_tax_metadata(text):
    """
    Scans the chunk for legal identifiers to enrich the vector database metadata.
    This allows for hybrid searching (e.g., filtering by Section before semantic search).
    """
    metadata = {}
    
    # Identify Section references (e.g., Section 80C, Sec. 43B, u/s 154)
    section_match = re.search(r'(?i)(?:section|sec\.?|u/s)\s*([0-9a-z]+)', text)
    if section_match:
        metadata['section'] = section_match.group(1).upper()
        
    # Identify Assessment Year (e.g., AY 2023-24, A.Y. 2024-25)
    ay_match = re.search(r'(?i)(?:a\.?y\.?|assessment year)\s*([0-9]{4}-[0-9]{2,4})', text)
    if ay_match:
        metadata['assessment_year'] = ay_match.group(1)
        
    # Identify common Indian Tax Acts
    if re.search(r'(?i)income\s?tax\s?act', text):
        metadata['act'] = 'Income Tax Act 1961'
    elif re.search(r'(?i)cgst|central goods and services', text):
        metadata['act'] = 'CGST Act 2017'
        
    return metadata

def build_intelligent_knowledge_base(input_folder, output_json_path):
    knowledge_base = []
    
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Created '{input_folder}' directory. Please place your tax PDFs inside it.")
        return

    # Context-Aware Chunking: 
    # This prioritises splitting by double newlines (paragraphs), then single newlines, 
    # then full stops, ensuring legal clauses aren't sliced in half.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    print(f"Scanning '{input_folder}' for tax documents...")
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing: {filename}")
            
            try:
                # Advanced Parsing: PyMuPDF respects layouts and reads tables much better
                doc = fitz.open(file_path)
                full_text = ""
                
                for page in doc:
                    # Extracting text while maintaining block structure
                    full_text += page.get_text("text") + "\n\n"
                
                # Create semantic chunks
                chunks = text_splitter.split_text(full_text)
                
                for index, chunk in enumerate(chunks):
                    # Enrich the chunk with searchable legal metadata
                    chunk_metadata = extract_tax_metadata(chunk)
                    
                    knowledge_base.append({
                        "document_name": filename,
                        "chunk_id": f"{filename}_chunk_{index}",
                        "content": chunk.strip(),
                        "metadata": chunk_metadata
                    })
                    
                doc.close()
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Export to JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=4, ensure_ascii=False)
        
    print(f"\nKnowledge base compilation complete. Saved {len(knowledge_base)} intelligent chunks to '{output_json_path}'.")

if __name__ == "__main__":
    build_intelligent_knowledge_base(input_folder="tax_pdfs", output_json_path="rag_knowledge_base.json")