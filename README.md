# Legalizer - AI Legal Document Assistant

Legalizer is an AI-powered legal document assistant that helps users analyze legal documents and ask questions about their content. Built with Streamlit, Google Gemini AI, and LangChain, it uses RAG (Retrieval-Augmented Generation) to provide accurate answers based on uploaded legal documents.

## Features

- Upload PDF legal documents for analysis
- Ask questions about the content of your documents
- Get AI-powered answers with source references
- Fallback mechanisms for handling API quotas
- Clean, intuitive user interface

## How It Works

1. Upload a legal PDF document
2. The system processes and splits the document into manageable chunks
3. Creates embeddings and stores them in a vector database
4. Ask questions about your document
5. Get AI-generated answers based on the document content

## Requirements

- Python 3.8+
- Google Gemini API key
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Mohitkoli987/legalizer.git
   ```

2. Navigate to the project directory:
   ```bash
   cd legalizer
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Google Gemini API key:
   - Create a `.env` file in the project root
   - Add your API key: `GEMINI_API_KEY=your_actual_api_key_here`

## Usage

Run the application:
```bash
streamlit run app.py
```

Open your browser to the URL provided (typically http://localhost:8501) and start using the legal assistant!

## Architecture

- **Frontend**: Streamlit for the web interface
- **AI Models**: Google Gemini for natural language processing
- **Document Processing**: PyPDF2 for PDF extraction
- **Text Splitting**: LangChain's RecursiveCharacterTextSplitter
- **Embeddings**: Google Generative AI Embeddings
- **Vector Storage**: ChromaDB
- **RAG Implementation**: LangChain RetrievalQA

## Disclaimer

This tool provides general information only and does not constitute legal advice. Always consult with a qualified attorney for legal matters.
