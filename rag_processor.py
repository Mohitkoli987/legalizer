import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from typing import List, Tuple

class LegalRAGProcessor:
    def __init__(self, api_key):
        self.api_key = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # Initialize the embedding model
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
        except Exception as e:
            print(f"Warning: Could not initialize embeddings model: {e}")
            self.embeddings = None
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.3,
            max_tokens=1024
        )
        
        # Initialize vector store
        self.vector_store = None
        self.chunks = []
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def split_text_into_chunks(self, text, chunk_size=1000, chunk_overlap=200):
        """Split text into chunks for better processing"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
    def create_vector_store(self, chunks):
        """Create vector store from text chunks"""
        if not self.embeddings:
            raise ValueError("Embeddings model not available due to quota limits")
            
        # Convert chunks to Document objects
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        try:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            return self.vector_store
        except Exception as e:
            if "quota" in str(e).lower():
                raise ValueError("Embedding quota exceeded. Please check your Google AI Studio quota or try again later.")
            else:
                raise e
    
    def process_document(self, pdf_path):
        """Process PDF document and create vector store"""
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Split into chunks
        chunks = self.split_text_into_chunks(text)
        self.chunks = chunks
        
        # Create vector store
        try:
            vector_store = self.create_vector_store(chunks)
            return vector_store, len(chunks)
        except ValueError as e:
            if "quota" in str(e).lower():
                # Fallback: Store chunks without embeddings
                print("Using fallback method due to quota limits")
                return None, len(chunks)
            else:
                raise e
    
    def create_qa_chain(self):
        """Create QA chain for question answering"""
        if not self.vector_store:
            # Fallback for quota issues
            raise ValueError("Vector store not initialized due to quota limits. Using fallback method.")
        
        # Create custom prompt template with better formatting instructions
        prompt_template = """
        You are a legal expert AI assistant. Answer the question using the provided context.
        Format your response as follows:
        1. Provide a concise and direct answer first.
        2. Highlight key terms such as names, dates, clauses, and parties in **bold**.
        3. Include the source information.
        4. End with the disclaimer.
        
        Context: {context}
        
        Question: {question}
        
        Answer in this exact format:
        Answer: [Concise answer]
        
        Key Terms: [Bold important terms]
        
        Source: [Approximate source in document]
        
        Disclaimer: Information provided is based on the uploaded document only and is for reference. Not legal advice.
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain
    
    def ask_question_with_fallback(self, question):
        """Ask a question with fallback for quota issues"""
        if self.vector_store is None:
            # Fallback method: Search through chunks manually
            relevant_chunks = []
            # Simple keyword matching as fallback
            question_lower = question.lower()
            for i, chunk in enumerate(self.chunks[:5]):  # Check first 5 chunks
                if any(word in chunk.lower() for word in question_lower.split()):
                    relevant_chunks.append(Document(page_content=chunk))
            
            # If no matches found, use first 3 chunks
            if not relevant_chunks:
                relevant_chunks = [Document(page_content=chunk) for chunk in self.chunks[:3]]
            
            # Create context from relevant chunks
            context = "\n\n".join([doc.page_content for doc in relevant_chunks])
            
            # Generate response using context
            prompt = f"""
            You are a legal expert AI assistant. Answer the question using the provided context.
            Format your response as follows:
            1. Provide a concise and direct answer first.
            2. Highlight key terms such as names, dates, clauses, and parties in **bold**.
            3. Include the source information.
            4. End with the disclaimer.
            
            Context: {context}
            
            Question: {question}
            
            Answer in this exact format:
            Answer: [Concise answer]
            
            Key Terms: [Bold important terms]
            
            Source: [Approximate source in document]
            
            Disclaimer: Information provided is based on the uploaded document only and is for reference. Not legal advice.
            """
            
            try:
                response = self.llm.invoke(prompt)
                # If the model doesn't follow the format, we'll format it properly
                content = response.content
                if not content.startswith("Answer:"):
                    # Try to extract key information and format it properly
                    formatted_response = f"Answer: {content}\n\n"
                    formatted_response += "Key Terms: [Key terms not specifically identified]\n\n"
                    formatted_response += "Source: [Approximate location in document]\n\n"
                    formatted_response += "Disclaimer: Information provided is based on the uploaded document only and is for reference. Not legal advice."
                    return {
                        "result": formatted_response,
                        "source_documents": relevant_chunks
                    }
                return {
                    "result": content,
                    "source_documents": relevant_chunks
                }
            except Exception as e:
                # Fallback response format if LLM fails
                fallback_response = f"Answer: Based on the document content, I can provide general information related to your question.\n\n"
                fallback_response += f"Key Terms: **{question}**, **document content**\n\n"
                fallback_response += "Source: Document content chunks\n\n"
                fallback_response += "Disclaimer: Information provided is based on the uploaded document only and is for reference. Not legal advice."
                return {
                    "result": fallback_response,
                    "source_documents": relevant_chunks
                }
        else:
            # Normal method with vector store
            qa_chain = self.create_qa_chain()
            result = qa_chain({"query": question})
            return result
    
    def ask_question(self, question, qa_chain=None):
        """Ask a question about the processed document"""
        try:
            if not qa_chain:
                result = self.ask_question_with_fallback(question)
            else:
                result = qa_chain({"query": question})
            return result
        except Exception as e:
            raise ValueError(f"Error processing question: {e}")