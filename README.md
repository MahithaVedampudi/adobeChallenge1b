Persona-Driven Document Intelligence System

Problem Statement
This system addresses the challenge of extracting and prioritizing relevant information from document collections based on specific user personas and their tasks. Traditional document processing systems treat all users equally, while this solution provides personalized content extraction tailored to:
- Different professional roles (HR, Travel Planner, etc.)
- Specific tasks/jobs-to-be-done
- Variable document types and domains

Technologies Used
 Python  Main programming language  3.9 
 Sentence Transformers  Text embeddings and similarity  2.2.2
 PyPDF2  PDF text extraction  3.0.1
 scikit-learn  Cosine similarity calculations  1.3.0 

Infrastructure
Docker Containerization and deployment 
JSON  Input/Output format 

Processing Pipeline
1. Input Handling:
   - Reads JSON configuration specifying persona, task, and documents
   - Validates file paths and document availability

2. Content Extraction:
   - Extracts text from PDFs with page-level precision
   - Identifies meaningful sections using layout analysis

3. Persona-Aware Ranking:
   - Generates embeddings for both content and task description
   - Ranks sections by semantic relevance using cosine similarity

4. Output Generation:
   - Produces structured JSON with ranked sections
   - Includes metadata and processing statistics

Setup Instructions:
docker build -t document-analysis .
docker run -v "C:\Users\mahit\document-analysis:/app/document-analysis" document-analysis
