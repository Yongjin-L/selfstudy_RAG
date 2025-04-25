# Self-Study Assistant - Streamlit App

**An AI-powered document question answering application using RAG (Retrieval Augmented Generation).**

This Streamlit application allows you to upload PDF documents and receive AI-powered answers based on the document content. It leverages a Retrieval Augmented Generation (RAG) pipeline for efficient and accurate responses.

## Features

*   **PDF Document Upload and Processing:** Upload PDF documents directly into the application.
*   **AI-Powered Question Answering:** Ask questions about the document content and receive intelligent answers.
*   **Vector Database Storage:** Documents are stored efficiently in a vector database.
*   **PDF Page Visualization:** View relevant PDF pages for context and reference.
*   **Interactive User Interface:**  A split-panel design for seamless question input and answer viewing.

## Requirements

*   Python 3.7+
*   OpenAI API Key
*   Streamlit
*   LangChain
*   LangChain-OpenAI (or LangChain-Community for older versions)
*   LangChain-Community
*   FAISS vector database
*   PyMuPDF (fitz)
*   dotenv

## Installation

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/yourusername/selfstudy-assistant.git
    cd selfstudy-assistant
    ```

2.  **Install required packages:**
    ```bash
    pip install streamlit langchain langchain-openai langchain-community faiss-cpu pymupdf python-dotenv
    ```

3.  **Create a `.env` file in the root directory:**
    ```
    OPENAI_API_KEY=your_api_key_here
    ```
    *   Replace `your_api_key_here` with your actual OpenAI API key.

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Access the application in your web browser** (typically at `http://localhost:8501`).

3.  **Upload a PDF document** using the file uploader and click "PDF Upload".

4.  **Ask questions about the document content** in the text input field.

5.  **View AI-generated answers and related document sections.**

6.  **Click on reference buttons to view the specific PDF pages** containing the information.

## How It Works

**Document Processing Pipeline:**

1.  **PDF Upload:** The application accepts PDF uploads and stores them temporarily.
2.  **Document Conversion:** PDFs are converted to text documents.
3.  **Chunking:** Documents are split into smaller chunks for better processing.
4.  **Vector Embedding:** Document chunks are embedded and stored in a FAISS vector database.

**Question Answering:**

1.  **Retrieval:** When a user asks a question, the system retrieves the most relevant document chunks.
2.  **Context Formation:** Retrieved chunks form the context for the LLM.
3.  **RAG Generation:** GPT-4o-mini generates answers based on the question and context.

**PDF Visualization:**

1.  **PDF to Image Conversion:** PDFs are converted to images for display.
2.  **Page Navigation:** Users can view specific pages referenced in the answers.

## Project Structure

*   `PDF Processing`: Functions for handling PDF uploads and conversion.
*   `Vector Database`: FAISS integration for document storage and retrieval.
*   `RAG Implementation`: Custom prompt template and query processing.
*   `UI Components`: Streamlit interface with split-panel design.

## Customization

*   Adjust chunk size and overlap in the `chunk_documents` function.
*   Modify the RAG prompt template in the `get_rag_chain` function.
*   Change the OpenAI models used for embeddings and generation.
*   Customize UI layout and styling through Streamlit components.

## License

MIT License

Copyright (c) 2025 Yongjin Lee
