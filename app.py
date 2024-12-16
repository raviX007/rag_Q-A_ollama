import streamlit as st
import os
import time
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings  # Changed from OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

def initialize_rag_system(url, groq_api_key, openai_api_key):
    """
    Initialize the Retrieval-Augmented Generation (RAG) system.
    
    Args:
        url (str): URL to load documents from
        groq_api_key (str): Groq API key for authentication
        openai_api_key (str): OpenAI API key for embeddings
    
    Returns:
        tuple: Initialized embeddings, vector store, and retrieval chain
    """
    try:
        # Validate API keys
        if not groq_api_key:
            st.error("Please enter your Groq API key")
            return None, None, None
        
        if not openai_api_key:
            st.error("Please enter your OpenAI API key")
            return None, None, None
        
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768"
        )
        
        # Initialize Embeddings with OpenAI
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Load and split documents
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        final_documents = text_splitter.split_documents(docs[:50])
        
        # Create vector store
        vector_store = FAISS.from_documents(final_documents, embeddings)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        </context>
        Question: {input}
        """)
        
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create retrieval chain
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return embeddings, vector_store, retrieval_chain
    
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None, None, None

def main():
    """
    Streamlit main application for RAG-based Q&A.
    """
    # Page configuration
    st.set_page_config(page_title="Groq RAG Q&A", page_icon="🤖")
    
    # Title and description
    st.title("🔍 Retrieval-Augmented Generation with Groq")
    st.markdown("Ask questions about documents using Groq and OpenAI!")
    
    # Sidebar for Configuration
    st.sidebar.header("🔧 RAG Configuration")
    
    # Groq API Key Input
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key. Get one at https://console.groq.com"
    )
    
    # OpenAI API Key Input
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key. Get one at https://platform.openai.com"
    )
    
    # Store API keys in session state if provided
    if groq_api_key:
        st.session_state.groq_api_key = groq_api_key
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
    
    # URL Input
    url = st.sidebar.text_input(
        "Document URL", 
        value="https://docs.smith.langchain.com/",
        help="Enter a URL to load documents from"
    )
    
    # Initialize RAG System Button
    if st.sidebar.button("Initialize RAG System"):
        # Check for API keys
        if not groq_api_key or not openai_api_key:
            st.sidebar.error("Please enter both Groq and OpenAI API keys!")
            return
            
        # Initialize the RAG system
        embeddings, vector_store, retrieval_chain = initialize_rag_system(url, groq_api_key, openai_api_key)
        
        # Store in session state
        if retrieval_chain:
            st.session_state.embeddings = embeddings
            st.session_state.vector_store = vector_store
            st.session_state.retrieval_chain = retrieval_chain
            st.sidebar.success("RAG system initialized successfully!")
    
    # Question Input and Processing
    prompt = st.text_input("Enter your question:")
    
    # Process Question
    if prompt:
        # Check if RAG system is initialized
        if 'retrieval_chain' not in st.session_state:
            st.warning("Please initialize the RAG system first!")
            return
        
        try:
            # Start timing
            start_time = time.time()
            
            # Invoke retrieval chain
            with st.spinner("Generating response..."):
                response = st.session_state.retrieval_chain.invoke({"input": prompt})
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Display Results
            st.subheader("Answer:")
            st.write(response['answer'])
            
            # Display Response Time
            st.info(f"Response Time: {response_time:.2f} seconds")
            
            # Display Retrieved Context
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"], 1):
                    st.markdown(f"*Document Snippet {i}:*")
                    st.write(doc.page_content)
                    st.write("---")
        
        except Exception as e:
            st.error(f"Error processing question: {e}")

if __name__ == "__main__":
    # User Agent setting
    os.environ["USER_AGENT"] = "my-streamlit-app"
    
    main()