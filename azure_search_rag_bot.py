
import os
import sys
from typing import List, Dict, Any, Set
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Ensure the current directory is in the path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from azure_search_rag import (
    verify_configuration,
    InferenceEngine,
    AzureSearchConfig
)

class AzureSearchRAG:
    """
    RAG System that uses Azure AI Search with multiple retrieval strategies
    to generate grounded answers.
    """
    
    def __init__(self):
        """Initialize the RAG system."""
        # 1. Load Configuration
        self.config = verify_configuration()
        
        # 2. Initialize Inference Engine (Retrieval)
        self.engine = InferenceEngine(self.config)
        
        # 3. Initialize LLM (Generation)
        self.llm = ChatOpenAI(
            api_key=self.config.openai_api_key,
            model_name="gpt-4.1-nano", # Or gpt-3.5-turbo, dependent on availability
            temperature=0.4
        )
        
        # 4. Define Generation Prompt
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant for the Azure AI Search RAG system.
        Answer the user's question using ONLY the context provided below.
        
        The context is aggregated from three different search strategies:
        1. Keyword Search
        2. Vector Hybrid Search
        3. Semantic Hybrid Search
        
        If the answer is not in the context, say "I don't have enough information to answer that."
        
        Context:
        {context}
        
        Question: 
        {question}
        
        Answer:
        """)

    def retrieve_comprehensive_context(self, query: str, k_per_strategy: int = 3) -> str:
        """
        Retrieve context using all three strategies and deduplicate results.
        
        Args:
            query: User question.
            k_per_strategy: Number of docs to retrieve per strategy.
            
        Returns:
            Aggregated string context.
        """
        strategies = ["keyword", "hybrid", "semantic"]
        all_results = []
        seen_content_hashes: Set[int] = set()
        
        print(f"\n🔍 Retrieving context for: '{query}'")
        
        for strategy in strategies:
            print(f"   - Running {strategy} search...")
            results = self.engine.query(query, strategy=strategy, k=k_per_strategy)
            
            for res in results:
                content = res['content']
                # Create a simple hash to deduplicate exact content matches
                content_hash = hash(content)
                
                if content_hash not in seen_content_hashes:
                    seen_content_hashes.add(content_hash)
                    # Add strategy metadata for transparency
                    res['_source_strategy'] = strategy
                    all_results.append(res)
        
        print(f"   => Total unique documents retrieved: {len(all_results)}")
        
        # Check if we have any results
        if not all_results:
            return ""
            
        # Format context
        context_parts = []
        for i, res in enumerate(all_results, 1):
            source = os.path.basename(res['metadata'].get('source', 'unknown'))
            strategy_used = res.get('_source_strategy', 'unknown')
            content_snippet = res['content']
            context_parts.append(f"--- Document {i} (Source: {source}, Strategy: {strategy_used}) ---\n{content_snippet}\n")
            
        return "\n".join(context_parts)

    def generate_answer(self, query: str):
        """
        Generate an answer for the query using the RAG pipeline.
        """
        # 1. Retrieve Context
        context = self.retrieve_comprehensive_context(query)
        
        if not context:
            return "No relevant information found in the knowledge base to answer your question."
        
        # 2. Generate Answer
        print("🤖 Generating answer...")
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = chain.invoke(query)
        return response

def main():
    rag = AzureSearchRAG()
    
    # Interactive Loop
    print("\n" + "="*60)
    print("      Welcome to Azure Search RAG (All-Strategies)")
    print("="*60)
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        user_input = input("\nEnter your question: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_input:
            continue
            
        try:
            answer = rag.generate_answer(user_input)
            print("\n" + "-"*60)
            print("ANSWER:")
            print(answer)
            print("-"*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
