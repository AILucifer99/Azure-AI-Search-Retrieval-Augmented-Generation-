"""
Document indexing operations for Azure Search.

This module provides functionality for indexing documents
to Azure Search with embeddings and metadata.
"""

from typing import List
import hashlib
import json
from langchain_core.documents import Document
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential


def index_documents_to_azure(
    documents: List[Document],
    embeddings,
    endpoint: str,
    index_name: str,
    key: str,
    batch_size: int = 100
) -> bool:
    """
    Index documents to Azure Search with embeddings.
    
    Args:
        documents: List of Document objects to index.
        embeddings: Embeddings model for generating vectors.
        endpoint: Azure Search service endpoint.
        index_name: Name of the target index.
        key: Azure Search admin key.
        batch_size: Number of documents to upload per batch (default: 100).
        
    Returns:
        True if indexing succeeded, False otherwise.
    """
    print(f"\n📊 Indexing {len(documents)} documents with proper field mapping...")
    
    search_client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(key)
    )
    
    search_documents = []
    
    # Process each document
    for i, doc in enumerate(documents):
        try:
            # Generate embedding
            print(f"   Processing document {i+1}/{len(documents)}...", end='\r')
            embedding = embeddings.embed_query(doc.page_content)
            
            # Generate unique ID
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 0)
            chunk = doc.metadata.get('chunk_id', 0)
            doc_id = hashlib.md5(f"{source}_{page}_{chunk}".encode()).hexdigest()
            
            # Map document fields
            search_doc = {
                "id": doc_id,
                "content": doc.page_content,
                "content_vector": embedding,
                "metadata": json.dumps(doc.metadata),
                
                # Explicitly map each field with defaults
                "source": str(doc.metadata.get("source", "")),
                "page": int(doc.metadata.get("page", 0)),
                "title": str(doc.metadata.get("title", "")),
                "author": str(doc.metadata.get("author", "")),
                "chunk_id": int(doc.metadata.get("chunk_id", 0)),
            }
            
            search_documents.append(search_doc)
        
        except Exception as e:
            print(f"\n   ⚠️  Error processing document {i}: {e}")
            continue
    
    print(f"\n   Prepared {len(search_documents)} documents for upload")
    
    # Upload in batches
    total_uploaded = 0
    
    for i in range(0, len(search_documents), batch_size):
        batch = search_documents[i:i+batch_size]
        try:
            result = search_client.upload_documents(documents=batch)
            successful = sum(1 for r in result if r.succeeded)
            failed = len(batch) - successful
            total_uploaded += successful
            
            print(f"   ✓ Batch {i//batch_size + 1}: {successful} succeeded, {failed} failed")
            
            # Show failures if any
            for r in result:
                if not r.succeeded:
                    print(f"      ❌ Failed: {r.key} - {r.error_message}")
        
        except Exception as e:
            print(f"   ❌ Error uploading batch: {e}")
    
    print(f"   ✅ Total uploaded: {total_uploaded}/{len(documents)}")
    
    return total_uploaded > 0


def generate_document_id(source: str, page: int, chunk_id: int) -> str:
    """
    Generate a unique document ID.
    
    Args:
        source: Source file path.
        page: Page number.
        chunk_id: Chunk identifier.
        
    Returns:
        MD5 hash as document ID.
    """
    return hashlib.md5(f"{source}_{page}_{chunk_id}".encode()).hexdigest()
