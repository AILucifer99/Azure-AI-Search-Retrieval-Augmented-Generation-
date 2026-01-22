"""
Inference Engine for Azure Search RAG.

This module provides a high-level facade for executing queries
using different search strategies.
"""

from typing import List, Literal, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

from .config import AzureSearchConfig
from .retrievers import (
    SearchStrategy,
    KeywordSearchStrategy,
    HybridSearchStrategy,
    SemanticHybridSearchStrategy
)

SearchType = Literal["keyword", "hybrid", "semantic"]

class InferenceEngine:
    """
    High-level engine for querying the Azure Search index.
    
    Abstracts away the complexity of switching between differnet
    retrieval strategies.
    """
    
    def __init__(self, config: AzureSearchConfig):
        """
        Initialize the inference engine.
        
        Args:
            config: Validated AzureSearchConfig object.
        """
        self.config = config
        self._embeddings = None
        self._vector_store = None
        self._strategies: Dict[str, SearchStrategy] = {}
        
        # Initialize resources lazily
        self._init_resources()
        
    def _init_resources(self):
        """Initialize Azure and OpenAI resources."""
        # Keyword Strategy (Does not need embeddings)
        self._strategies["keyword"] = KeywordSearchStrategy(
            endpoint=self.config.azure_search_endpoint,
            key=self.config.azure_search_key,
            index_name=self.config.azure_search_index_name
        )
        
        # Embeddings & Vector Store (Needed for Hybrid & Semantic)
        self._embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key,
            model=self.config.embedding_model
        )
        
        self._vector_store = AzureSearch(
            azure_search_endpoint=self.config.azure_search_endpoint,
            azure_search_key=self.config.azure_search_key,
            index_name=self.config.azure_search_index_name,
            embedding_function=self._embeddings.embed_query,
            semantic_configuration_name="my-semantic-config"
        )
        
        self._strategies["hybrid"] = HybridSearchStrategy(self._vector_store)
        self._strategies["semantic"] = SemanticHybridSearchStrategy(self._vector_store)
        
    def query(self, query: str, strategy: SearchType = "hybrid", k: int = 5) -> List[Dict[str, Any]]:
        """
        Execute a query using the specified strategy.
        
        Args:
            query: Search query text.
            strategy: One of 'keyword', 'hybrid', 'semantic'.
            k: Number of results to retrieve.
            
        Returns:
            List of results (dictionaries with content and metadata).
        """
        if strategy not in self._strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Valid options: {list(self._strategies.keys())}")
            
        search_strategy = self._strategies[strategy]
        documents = search_strategy.search(query, k=k)
        
        # Standardize output format for easy consumption
        results = []
        for doc in documents:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score") or doc.metadata.get("@search.score"),
                "strategy": strategy
            })
            
        return results

    def get_available_strategies(self) -> List[str]:
        """Return list of available search strategies."""
        return list(self._strategies.keys())
