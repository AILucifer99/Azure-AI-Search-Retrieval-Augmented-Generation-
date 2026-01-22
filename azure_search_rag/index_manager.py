"""
Azure Search index management.

This module provides functionality for creating and managing
Azure Search indexes with vector and semantic search capabilities.
"""

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    VectorSearchAlgorithmKind,
)
from azure.core.credentials import AzureKeyCredential


class AzureSearchIndexManager:
    """Manager for creating and managing Azure Search indexes."""
    
    def __init__(self, endpoint: str, key: str, index_name: str):
        """
        Initialize the index manager.
        
        Args:
            endpoint: Azure Search service endpoint.
            key: Azure Search admin key.
            index_name: Name of the index to manage.
        """
        self.endpoint = endpoint
        self.key = key
        self.index_name = index_name
        self.credential = AzureKeyCredential(self.key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
    
    def create_index_with_vector_and_semantic(self, embedding_dimensions: int = 1536) -> bool:
        """
        Create an Azure Search index with vector and semantic search capabilities.
        
        Args:
            embedding_dimensions: Dimension of the embedding vectors (default: 1536).
            
        Returns:
            True if index creation succeeded, False otherwise.
        """
        print(f"\n🔧 Creating Azure Search Index: {self.index_name}")
        
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="en.microsoft",
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=embedding_dimensions,
                vector_search_profile_name="myHnswProfile",
            ),
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SimpleField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SimpleField(
                name="page",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
                facetable=True,
            ),
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
            ),
            SearchableField(
                name="author",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
                facetable=True,
            ),
            SimpleField(
                name="chunk_id",
                type=SearchFieldDataType.Int32,
                filterable=True,
            ),
        ]
        
        # Configure vector search with HNSW algorithm
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine",
                    },
                ),
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                ),
            ],
        )
        
        # Configure semantic search
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")],
                keywords_fields=[
                    SemanticField(field_name="metadata"),
                    SemanticField(field_name="author"),
                ],
            ),
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        # Create index definition
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
        
        try:
            # Delete existing index if it exists
            try:
                self.index_client.delete_index(self.index_name)
                print(f"   ✓ Deleted existing index")
            except:
                pass
            
            # Create new index
            result = self.index_client.create_index(index)
            print(f"   ✓ Created index: {result.name}")
            print(f"   ✓ Vector search configured with HNSW")
            print(f"   ✓ Semantic search configured")
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating index: {e}")
            return False
    
    def check_index_exists(self) -> bool:
        """
        Check if the index exists.
        
        Returns:
            True if index exists, False otherwise.
        """
        try:
            self.index_client.get_index(self.index_name)
            return True
        except:
            return False
    
    def delete_index(self) -> bool:
        """
        Delete the index.
        
        Returns:
            True if deletion succeeded, False otherwise.
        """
        try:
            self.index_client.delete_index(self.index_name)
            print(f"   ✓ Deleted index: {self.index_name}")
            return True
        except Exception as e:
            print(f"   ❌ Error deleting index: {e}")
            return False
