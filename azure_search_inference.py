"""
Azure Search Inference - Demo Script

This script demonstrates how to use the modular inference engine
to perform Keyword, Hybrid, and Semantic Hybrid searches.
Results are saved to 'inference_results.txt'.
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# Import from our modular package
from azure_search_rag import (
    verify_configuration,
    InferenceEngine
)


class ResultLogger:
    """Helper class to log output to both console and file."""
    
    def __init__(self, filename="inference_results.txt"):
        self.filename = filename
        # Clear/Create file
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(f"Azure Search Inference Results\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
    def log(self, message=""):
        """Print to console and append to file."""
        print(message)
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(message + "\n")


def format_results(results, title, logger):
    """Helper to pretty-print search results."""
    logger.log(f"\n{title}")
    logger.log("=" * 80)
    
    if not results:
        logger.log("No results found.")
        return
        
    for i, res in enumerate(results, 1):
        score = res.get('score', 'N/A')
        file_name = os.path.basename(res['metadata'].get('source', 'Unknown'))
        page = res['metadata'].get('page', '?')
        content_snippet = res['content'][:200].replace('\n', ' ')
        
        logger.log(f"[{i}] Score: {score} | File: {file_name} (Page {page})")
        logger.log(f"    Content: {content_snippet}...")
        logger.log("-" * 50)


def main():
    # 0. Initialize Logger
    logger = ResultLogger("inference_results.txt")
    logger.log("Initializing Inference Engine...")
    
    # 1. Load and verify config
    config = verify_configuration()
    
    # 2. Initialize Engine
    engine = InferenceEngine(config)
    
    logger.log(f"\nAvailable strategies: {engine.get_available_strategies()}")
    
    # 3. Define Queries
    queries = [
        "What are the main components of a RAG system?",
        "Explain the difference between sparse and dense retrieval",
        "How does semantic reranking improve results?"
    ]
    
    # 4. Run Inference Loop
    for query in queries:
        logger.log(f"\n\n{'#'*80}")
        logger.log(f"🔍 QUERY: {query}")
        logger.log(f"{'#'*80}")
        
        # A. Keyword Search
        keyword_results = engine.query(query, strategy="keyword", k=3)
        format_results(keyword_results, "📝 KEYWORD SEARCH (BM25)", logger)
        
        # B. Hybrid Search
        hybrid_results = engine.query(query, strategy="hybrid", k=3)
        format_results(hybrid_results, "⚡ HYBRID SEARCH (Vector + Keyword)", logger)
        
        # C. Semantic Hybrid Search
        semantic_results = engine.query(query, strategy="semantic", k=3)
        format_results(semantic_results, "🧠 SEMANTIC HYBRID SEARCH (Reranked)", logger)
        
    print(f"\n\n✅ Results saved to {logger.filename}")
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
