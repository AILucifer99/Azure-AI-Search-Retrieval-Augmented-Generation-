"""
Retrieval strategies for Azure Search RAG system.

This module defines various strategies for retrieving documents
from Azure Search, including Keyword, Hybrid, and Semantic Hybrid search.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import json

from langchain_core.documents import Document
from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""
    
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Execute search query.
        
        Args:
            query: Search query string.
            k: Number of documents to retrieve.
            
        Returns:
            List of retrieved documents.
        """
        pass


class KeywordSearchStrategy(SearchStrategy):
    """
    Keyword-only search strategy using Azure Search.
    Uses standard BM25 ranking without vector embeddings.
    """
    
    def __init__(self, endpoint: str, key: str, index_name: str):
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(key)
        )
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        results = self.client.search(
            search_text=query,
            top=k,
            include_total_count=True,
            select=["content", "metadata", "id", "source", "page", "title", "author", "chunk_id"]
        )
        
        documents = []
        for result in results:
            metadata = self._parse_metadata(result)
            metadata["score"] = result.get("@search.score", 0)
            metadata["search_type"] = "keyword"
            
            documents.append(Document(
                page_content=result.get("content", ""),
                metadata=metadata
            ))
        return documents

    def _parse_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and parse metadata from result."""
        metadata = {}
        # Try to parse JSON formatted metadata field
        if result.get("metadata"):
            try:
                metadata = json.loads(result["metadata"])
            except:
                pass
        
        # Override/Ensure top-level fields
        for field in ["source", "page", "title", "author", "chunk_id"]:
            if val := result.get(field):
                metadata[field] = val
                
        return metadata


class HybridSearchStrategy(SearchStrategy):
    """
    Hybrid search strategy (Keyword + Vector).
    Combines BM25 and vector similarity scores (RRF by default in Azure).
    """
    
    def __init__(self, vector_store: AzureSearch):
        self.vector_store = vector_store
        
    def search(self, query: str, k: int = 5) -> List[Document]:
        # AzureSearch.hybrid_search returns list of documents with metadata
        return self.vector_store.hybrid_search(
            query=query,
            k=k
        )


class SemanticHybridSearchStrategy(SearchStrategy):
    """
    Semantic Hybrid search strategy (Keyword + Vector + Semantic Reranking).
    Uses Azure's semantic ranker to improve relevance of top results.
    """
    
    def __init__(self, vector_store: AzureSearch):
        self.vector_store = vector_store
        
    def search(self, query: str, k: int = 5) -> List[Document]:
        # Perform semantic hybrid search
        return self.vector_store.semantic_hybrid_search(
            query=query,
            k=k
        )
