import os
import tempfile
import shutil
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from rag_processor import LegalRAGProcessor

# Load .env
load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env")
    st.stop()

# Configure Gemini
genai.configure(api_key=API_KEY)

# ✅ Use available model
MODEL_NAME = "gemini-2.0-flash"

try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Legal AI Chatbot", page_icon="⚖️")

st.title("⚖️ Legal AI Chatbot")
st.caption("General information only — not legal advice.")

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Session state for RAG processor
if "rag_processor" not in st.session_state:
    st.session_state.rag_processor = None

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "doc_chunk_count" not in st.session_state:
    st.session_state.doc_chunk_count = 0

if "processing_error" not in st.session_state:
    st.session_state.processing_error = None

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# File uploader for legal documents
st.sidebar.header("Legal Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a legal PDF document", type=["pdf"])

if uploaded_file is not None:
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Process the document
    with st.spinner("Processing document..."):
        try:
            st.session_state.rag_processor = LegalRAGProcessor(API_KEY)
            vector_store, chunk_count = st.session_state.rag_processor.process_document(tmp_file_path)
            if vector_store is not None:
                st.session_state.qa_chain = st.session_state.rag_processor.create_qa_chain()
            st.session_state.document_processed = True
            st.session_state.doc_chunk_count = chunk_count
            st.session_state.processing_error = None
            st.sidebar.success(f"Document processed successfully! ({chunk_count} chunks)")
        except Exception as e:
            st.session_state.processing_error = str(e)
            st.sidebar.error(f"Error processing document: {str(e)}")
            # Still mark as processed so we can use fallback
            st.session_state.document_processed = True
    
    # Clean up temporary file
    try:
        os.unlink(tmp_file_path)
    except:
        pass

# Display document info if processed
if st.session_state.document_processed:
    if st.session_state.processing_error:
        st.sidebar.warning(f"Document loaded with limitations: {st.session_state.processing_error}")
    else:
        st.sidebar.info(f"Document processed: {st.session_state.doc_chunk_count} chunks")

user_input = st.chat_input("Ask a legal question...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Use RAG if document is processed, otherwise use general legal assistant
    if st.session_state.document_processed and st.session_state.rag_processor:
        try:
            result = st.session_state.rag_processor.ask_question(user_input, st.session_state.qa_chain)
            assistant_text = result["result"]
            
            # Add source information
            if "Disclaimer:" not in assistant_text:
                source_info = "\n\n**Sources from document:**\n"
                for i, doc in enumerate(result["source_documents"]):
                    # Show first 200 characters of each source
                    content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    source_info += f"{i+1}. {content_preview}\n"
                
                assistant_text += source_info
        except Exception as e:
            assistant_text = f"❌ Error generating response: {e}"
    else:
        system_prompt = """
        You are a legal information assistant.
        Do not provide legal advice.
        Provide only general legal information.
        Always recommend consulting a licensed attorney.
        """
        
        try:
            response = model.generate_content(
                system_prompt + "\n\nUser Question: " + user_input
            )
            assistant_text = response.text
        except Exception as e:
            assistant_text = f"❌ Generation error: {e}"

    with st.chat_message("assistant"):
        st.markdown(assistant_text)
        # Only add general disclaimer if RAG response doesn't already have one
        if "Information provided is based on the uploaded document" not in assistant_text:
            st.markdown(
                "> ⚠️ **Disclaimer:** This is general information only and not legal advice. "
                "Consult a licensed attorney."
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_text}
    )