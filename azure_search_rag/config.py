"""
Configuration management for Azure Search RAG system.

This module handles environment variable loading, validation,
and configuration settings.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AzureSearchConfig:
    """Configuration settings for Azure Search RAG system."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.azure_search_key = os.getenv("AZURE_SEARCH_KEY")
        self.azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    
    def validate(self) -> None:
        """
        Validate that all required configuration values are present.
        
        Raises:
            ValueError: If any required configuration is missing.
        """
        required_vars = {
            "OPENAI_API_KEY": self.openai_api_key,
            "AZURE_SEARCH_ENDPOINT": self.azure_search_endpoint,
            "AZURE_SEARCH_KEY": self.azure_search_key,
            "AZURE_SEARCH_INDEX_NAME": self.azure_search_index_name,
        }
        
        missing = [var for var, value in required_vars.items() if not value]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    def display(self) -> None:
        """Display configuration (masking sensitive values)."""
        print("🔍 Configuration:")
        print(f"   ✓ OPENAI_API_KEY: {'*' * 20}")
        print(f"   ✓ AZURE_SEARCH_ENDPOINT: {self.azure_search_endpoint}")
        print(f"   ✓ AZURE_SEARCH_KEY: {'*' * 20}")
        print(f"   ✓ AZURE_SEARCH_INDEX_NAME: {self.azure_search_index_name}")
        print(f"   ✓ EMBEDDING_MODEL: {self.embedding_model}")
        print(f"   ✓ EMBEDDING_DIMENSIONS: {self.embedding_dimensions}")


def verify_configuration() -> AzureSearchConfig:
    """
    Verify and return configuration.
    
    Returns:
        AzureSearchConfig: Validated configuration object.
        
    Raises:
        ValueError: If configuration validation fails.
    """
    print("🔍 Verifying Configuration...")
    config = AzureSearchConfig()
    config.validate()
    
    # Display masked configuration
    for var in ["OPENAI_API_KEY", "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY", "AZURE_SEARCH_INDEX_NAME"]:
        value = getattr(config, var.lower())
        display = '*' * 20 if 'KEY' in var else value
        print(f"   ✓ {var}: {display}")
    
    print("✅ Configuration verified!\n")
    return config
