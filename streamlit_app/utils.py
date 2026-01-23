import sys
import os
import streamlit as st

# Add parent directory to sys.path to allow imports from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from azure_search_rag_bot import AzureSearchRAG
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    # Fallback or error handling if run from wrong location
    st.error("Could not import logic from parent directory. Please ensure the app is run correctly.")
    AzureSearchRAG = None

def load_css(file_name):
    """Load custom CSS from a file."""
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def get_rag_system():
    """Initialize and cache the RAG system."""
    return AzureSearchRAG()

def format_source_display(context_str: str):
    """Parse the context string back into structured data for display."""
    if not context_str:
        return []
    
    # This is a simple parser based on the known format in azure_search_rag_bot.py
    # Format: --- Document {i} (Source: {source}, Strategy: {strategy_used}) ---\n{content}\n
    
    sources = []
    import re
    
    # Regex to find document headers
    # Example: --- Document 1 (Source: file.pdf, Strategy: keyword) ---
    pattern = r"--- Document \d+ \(Source: (.*?), Strategy: (.*?)\) ---\n(.*?)(?=\n--- Document|\Z)"
    matches = re.findall(pattern, context_str, re.DOTALL)
    
    for source, strategy, content in matches:
        sources.append({
            "source": source.strip(),
            "strategy": strategy.strip(),
            "content": content.strip()
        })
        
    return sources

def stream_response(rag, query, context):
    """Stream the response from the RAG system."""
    
    # Recreate the chain logic from AzureSearchRAG.generate_answer
    # We want to stream, so we use chain.stream()
    
    chain = (
        rag.prompt
        | rag.llm
        | StrOutputParser()
    )
    
    # Prepare the input for the prompt
    input_data = {
        "context": context,
        "question": query
    }
    
    return chain.stream(input_data)
