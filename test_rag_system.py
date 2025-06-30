#!/usr/bin/env python3
"""
Test RAG System
Builds RAG-ready graph and tests the RAG functionality.
"""

import os
import json
import logging
from rag_ready_graph_builder import RAGReadyGraphBuilder
from app.rag.rag_engine import RAGEngine
from app.core.knowledge_graph import KnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_system():
    """Test the complete RAG system."""
    
    print("=" * 60)
    print("TESTING RAG SYSTEM")
    print("=" * 60)
    
    # Step 1: Build RAG-ready graph
    print("\n1. Building RAG-ready graph...")
    try:
        builder = RAGReadyGraphBuilder()
        rag_structure = builder.build_rag_ready_graph("cholera", max_results=3)
        
        print(f"✓ RAG-ready graph built successfully!")
        print(f"  - Total chunks: {rag_structure['metadata']['total_chunks']}")
        print(f"  - Sources: {rag_structure['metadata']['chunks_by_source']}")
        
    except Exception as e:
        print(f"✗ Failed to build RAG-ready graph: {e}")
        return False
    
    # Step 2: Test RAG engine
    print("\n2. Testing RAG engine...")
    try:
        # Initialize knowledge graph and RAG engine
        kg = KnowledgeGraph()
        rag_engine = RAGEngine(kg)
        
        # Test questions
        test_questions = [
            "What are the symptoms of cholera?",
            "How is cholera treated?",
            "What drugs are used for cholera treatment?",
            "What is the prevalence of cholera?"
        ]
        
        for question in test_questions:
            print(f"\nQuestion: {question}")
            
            # Test RAG-ready method
            result = rag_engine.answer_question(question, use_rag_ready=True)
            
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Confidence: {result.get('confidence', 0.0):.2f}")
            print(f"Context sources: {len(result.get('context', []))}")
            
            if result.get('follow_up_question'):
                print(f"Follow-up: {result['follow_up_question']}")
        
        print("\n✓ RAG engine tests completed!")
        return True
        
    except Exception as e:
        print(f"✗ Failed to test RAG engine: {e}")
        return False

def test_enhanced_pubmed():
    """Test the enhanced PubMed collector."""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED PUBMED COLLECTOR")
    print("=" * 60)
    
    try:
        from app.data.enhanced_pubmed_collector import EnhancedPubMedCollector
        
        # Check if API keys are available
        if not (os.getenv("NCBI_API_KEY") and os.getenv("NCBI_EMAIL")):
            print("⚠️  NCBI_API_KEY and NCBI_EMAIL not set. Skipping PubMed test.")
            return True
        
        collector = EnhancedPubMedCollector()
        nodes = collector.search_and_fetch("cholera", max_results=2, include_fulltext=False)
        
        print(f"✓ Enhanced PubMed collector test successful!")
        print(f"  - Retrieved {len(nodes['nodes'])} articles")
        
        # Save test results
        collector.save_to_file(nodes, "test_pubmed_nodes.json")
        print(f"  - Saved to test_pubmed_nodes.json")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced PubMed collector test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Starting RAG system tests...")
    
    # Test enhanced PubMed collector
    pubmed_success = test_enhanced_pubmed()
    
    # Test RAG system
    rag_success = test_rag_system()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Enhanced PubMed Collector: {'✓ PASS' if pubmed_success else '✗ FAIL'}")
    print(f"RAG System: {'✓ PASS' if rag_success else '✗ FAIL'}")
    
    if rag_success:
        print("\n🎉 RAG system is working! You can now:")
        print("1. Use the web interface at http://localhost:8000")
        print("2. Build RAG-ready graphs for different diseases")
        print("3. Query the system with medical questions")
    else:
        print("\n❌ RAG system needs fixing. Check the error messages above.")

if __name__ == "__main__":
    main() 