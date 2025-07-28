import json
import os
import time
from typing import List, Dict, Any
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentIntelligenceSystem:
    def __init__(self):
        # Load the pre-trained model (within 1GB size limit)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with page numbers"""
        try:
            reader = PdfReader(pdf_path)
            return [
                {"page_number": i + 1, "text": page.extract_text() or ""}
                for i, page in enumerate(reader.pages)
            ]
        except Exception as e:
            print(f"Error reading {pdf_path}: {str(e)}")
            return []

    def process_collection(self, collection_path: str):
        """Process a single collection with improved error handling"""
        try:
            # Path configuration
            pdf_folder = os.path.join(collection_path, 'PDFs')
            input_file = os.path.join(collection_path, 'challenge1b_input.json')
            
            # Validate input file
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found at {input_file}")

            with open(input_file) as f:
                input_data = json.load(f)

            # Extract data with safety checks
            persona = input_data.get('persona', {}).get('role', 'Unknown')
            job = input_data.get('job_to_be_done', {}).get('task', 'Unknown')
            documents = [doc['filename'] for doc in input_data.get('documents', [])]

            if not documents:
                raise ValueError("No documents specified in input file")

            # Process all PDFs
            all_sections = []
            for doc in documents:
                pdf_path = os.path.join(pdf_folder, doc)
                if not os.path.exists(pdf_path):
                    print(f"Warning: PDF {doc} not found at {pdf_path}")
                    continue
                
                pages = self.extract_text_from_pdf(pdf_path)
                for page in pages:
                    if page['text'].strip():  # Only add non-empty pages
                        all_sections.append({
                            "document": doc,
                            "page_number": page['page_number'],
                            "text": page['text']
                        })

            if not all_sections:
                raise ValueError("No readable text found in any documents")

            # Rank sections by relevance
            ranked_sections = self.rank_sections(all_sections, persona, job)
            
            # Prepare output structure
            output = {
                "metadata": {
                    "input_documents": documents,
                    "persona": persona,
                    "job_to_be_done": job,
                    "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "documents_processed": len([s for s in all_sections]),
                    "documents_failed": len(documents) - len(set(s['document'] for s in all_sections))
                },
                "extracted_sections": [],
                "sub_section_analysis": []
            }

            # Add top sections to output (with better section titles)
            for i, section in enumerate(ranked_sections[:10]):
                section_title = self.extract_section_title(section['text'])
                output['extracted_sections'].append({
                    "document": section['document'],
                    "page_number": section['page_number'],
                    "section_title": section_title or f"Section {i+1}",
                    "importance_rank": i + 1
                })
                
                output['sub_section_analysis'].append({
                    "document": section['document'],
                    "page_number": section['page_number'],
                    "refined_text": self.refine_text(section['text']),
                    "relevance_score": float(section['similarity_score'])
                })

            return output

        except Exception as e:
            print(f"Error processing collection: {str(e)}")
            return None

    def extract_section_title(self, text: str) -> str:
        """Extract meaningful section title from text"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return ""
        
        # First non-empty line that's not too long
        for line in lines[:3]:
            if len(line) < 100 and not line.isdigit():
                return line
        return ""

    def rank_sections(self, sections: List[Dict], persona: str, job: str) -> List[Dict]:
        """Rank sections by relevance to persona and job"""
        try:
            persona_job_text = f"{persona} needs to {job}"
            persona_job_embedding = self.model.encode([persona_job_text])[0]
            section_texts = [s['text'] for s in sections]
            section_embeddings = self.model.encode(section_texts)
            
            similarities = cosine_similarity(
                [persona_job_embedding],
                section_embeddings
            )[0]
            
            ranked = []
            for i, section in enumerate(sections):
                ranked.append({
                    **section,
                    "similarity_score": similarities[i]
                })
            
            return sorted(ranked, key=lambda x: x['similarity_score'], reverse=True)
        except Exception as e:
            print(f"Ranking error: {str(e)}")
            return sections  # Return original order if ranking fails

    def refine_text(self, text: str, max_length: int = 300) -> str:
        """Refine text to be more concise while preserving key info"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return text[:max_length] + ('...' if len(text) > max_length else '')
            
        refined = sentences[0]
        for s in sentences[1:]:
            if len(refined) + len(s) + 1 <= max_length:
                refined += '. ' + s
            else:
                break
        return refined + ('...' if len(refined) < len(text) else '')

def main():
    system = DocumentIntelligenceSystem()
    base_dir = "/app/document-analysis"
    
    for collection in ['collection1', 'collection2', 'collection3']:
        collection_path = os.path.join(base_dir, 'input', collection)
        output_file = os.path.join(base_dir, collection, 'challenge1b_output.json')
        
        print(f"\nProcessing {collection}...")
        output = system.process_collection(collection_path)
        
        if output:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Successfully processed {collection}. Output saved to {output_file}")
        else:
            print(f"Failed to process {collection}")

if __name__ == "__main__":
    main()