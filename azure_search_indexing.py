"""
Azure Search Indexing - Main Entry Point

This script provides the main entry point for setting up Azure Search
indexes with PDF documents using the modular azure_search_rag package.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import from our modular package
from azure_search_rag import (
    verify_configuration,
    CustomPyMuPDFLoader,
    AzureSearchIndexManager,
    index_documents_to_azure,
)


def setup_azure_search_index(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Setup Azure Search index with PDF documents.
    
    Args:
        pdf_path: Path to the PDF file to index.
        chunk_size: Size of text chunks (default: 1000).
        chunk_overlap: Overlap between chunks (default: 200).
        
    Returns:
        True if setup succeeded, False otherwise.
    """
    print("="*80)
    print("🚀 SETTING UP AZURE SEARCH INDEX")
    print("="*80)
    
    # Step 1: Verify configuration
    config = verify_configuration()
    
    # Step 2: Initialize embeddings
    print("\n🔤 Initializing OpenAI Embeddings...")
    embeddings = OpenAIEmbeddings(
        openai_api_key=config.openai_api_key,
        model=config.embedding_model
    )
    print("   ✓ Embeddings initialized")
    
    # Step 3: Create/verify index
    print("\n🔍 Setting up Azure Search Index...")
    index_manager = AzureSearchIndexManager(
        endpoint=config.azure_search_endpoint,
        key=config.azure_search_key,
        index_name=config.azure_search_index_name
    )
    
    if not index_manager.check_index_exists():
        index_manager.create_index_with_vector_and_semantic(
            embedding_dimensions=config.embedding_dimensions
        )
    else:
        print(f"   Index exists. Recreate? (y/n): ", end='')
        if input().strip().lower() == 'y':
            index_manager.create_index_with_vector_and_semantic(
                embedding_dimensions=config.embedding_dimensions
            )
        else:
            print("   ✓ Using existing index")
    
    # Step 4: Load PDF
    print("\n📄 Loading PDF...")
    loader = CustomPyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    # Step 5: Split documents
    print(f"\n✂️  Splitting into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    
    # Add chunk metadata
    for i, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = i
        doc.metadata["total_chunks"] = len(split_docs)
    
    print(f"   ✓ Created {len(split_docs)} chunks")
    
    # Step 6: Index documents
    success = index_documents_to_azure(
        documents=split_docs,
        embeddings=embeddings,
        endpoint=config.azure_search_endpoint,
        index_name=config.azure_search_index_name,
        key=config.azure_search_key
    )
    
    if not success:
        raise Exception("Indexing failed")
    
    print("\n" + "="*80)
    print("✅ INDEX SETUP COMPLETE!")
    print("="*80)
    
    return True


def main():
    """Main entry point."""
    PDF_PATH = "Data\\RAG-Papers-2025\\REFRAG.pdf"
    
    try:
        setup_azure_search_index(PDF_PATH)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
