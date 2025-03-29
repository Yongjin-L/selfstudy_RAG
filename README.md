Self-Study Assistant
A Streamlit application that allows users to upload PDF documents, ask questions, and receive AI-powered answers based on the document content using RAG (Retrieval Augmented Generation).
Features

PDF document upload and processing
Vector database storage of document content
Question answering based on document context
PDF page visualization for reference
Interactive user interface with split-panel design

Requirements

Python 3.7+
OpenAI API key
Streamlit
LangChain
PyMuPDF (fitz)
FAISS vector database
dotenv

Installation

Clone this repository:

Copygit clone https://github.com/yourusername/selfstudy-assistant.git
cd selfstudy-assistant

Install required packages:

Copypip install streamlit langchain langchain-openai langchain-community faiss-cpu pymupdf python-dotenv

Create a .env file in the root directory with your OpenAI API key:

CopyOPENAI_API_KEY=your_api_key_here
Usage

Run the Streamlit application:

Copystreamlit run app.py

Access the application in your web browser (typically at http://localhost:8501)
Upload a PDF document using the file uploader and click "PDF Upload"
Ask questions about the document content in the text input field
View AI-generated answers and related document sections
Click on reference buttons to view the specific PDF pages containing the information

How It Works
Document Processing Pipeline

PDF Upload: The application accepts PDF uploads and stores them temporarily.
Document Conversion: PDFs are converted to text documents.
Chunking: Documents are split into smaller chunks for better processing.
Vector Embedding: Document chunks are embedded and stored in a FAISS vector database.

Question Answering

Retrieval: When a user asks a question, the system retrieves the most relevant document chunks.
Context Formation: Retrieved chunks form the context for the LLM.
RAG Generation: GPT-4o-mini generates answers based on the question and context.

PDF Visualization

PDF to Image Conversion: PDFs are converted to images for display.
Page Navigation: Users can view specific pages referenced in the answers.

Project Structure

PDF Processing: Functions for handling PDF uploads and conversion
Vector Database: FAISS integration for document storage and retrieval
RAG Implementation: Custom prompt template and query processing
UI Components: Streamlit interface with split-panel design

Customization

Adjust chunk size and overlap in the chunk_documents function
Modify the RAG prompt template in the get_rag_chain function
Change the OpenAI models used for embeddings and generation
Customize UI layout and styling through Streamlit components

License
MIT License
Copyright (c) 2025 Yongjin Lee
