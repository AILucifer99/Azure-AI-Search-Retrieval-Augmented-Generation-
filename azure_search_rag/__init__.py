"""
Azure Search RAG Package

A modular package for building RAG systems with Azure AI Search.
Provides PDF loading, index management, document indexing, and inference capabilities.
"""

from .config import AzureSearchConfig, verify_configuration
from .loaders import CustomPyMuPDFLoader
from .index_manager import AzureSearchIndexManager
from .indexer import index_documents_to_azure, generate_document_id

# Inference exports
from .engine import InferenceEngine
from .retrievers import (
    SearchStrategy,
    KeywordSearchStrategy,
    HybridSearchStrategy,
    SemanticHybridSearchStrategy
)

__version__ = "1.1.0"

__all__ = [
    # Configuration
    "AzureSearchConfig",
    "verify_configuration",
    
    # Loaders
    "CustomPyMuPDFLoader",
    
    # Index Management
    "AzureSearchIndexManager",
    
    # Indexing
    "index_documents_to_azure",
    "generate_document_id",
    
    # Inference
    "InferenceEngine",
    "SearchStrategy",
    "KeywordSearchStrategy",
    "HybridSearchStrategy",
    "SemanticHybridSearchStrategy",
]
