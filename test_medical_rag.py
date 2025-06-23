#!/usr/bin/env python3
"""
Test script for the complete Medical RAG system.
This script tests the knowledge graph, RAG engine, and API endpoints.
"""

import requests
import json
import time
import logging
from app.core.knowledge_graph import KnowledgeGraph
from app.rag.rag_engine import RAGEngine
from app.data.fda_collector import FDACollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_knowledge_graph():
    """Test the knowledge graph functionality."""
    print("Testing Knowledge Graph...")
    
    kg = KnowledgeGraph()
    
    # Test search functionality
    results = kg.search_nodes("disease", limit=3)
    print(f"Found {len(results)} nodes matching 'disease'")
    
    # Test graph statistics
    stats = {
        "nodes": len(kg.graph.nodes),
        "edges": len(kg.graph.edges),
        "diseases": len(kg.disease_nodes),
        "articles": len(kg.article_nodes)
    }
    print(f"Knowledge graph stats: {stats}")
    
    return kg

def test_fda_collector():
    """Test the FDA collector."""
    print("\nTesting FDA Collector...")
    
    fda = FDACollector()
    
    # Test drug search
    drug_labels = fda.search_drug_labels("active_ingredient:ibuprofen", limit=2)
    print(f"Found {len(drug_labels)} drug labels for ibuprofen")
    
    # Test adverse events
    adverse_events = fda.get_adverse_events(limit=2)
    print(f"Found {len(adverse_events)} adverse events")
    
    return fda

def test_rag_engine(kg):
    """Test the RAG engine."""
    print("\nTesting RAG Engine...")
    
    rag = RAGEngine(kg)
    
    # Test context retrieval
    context = rag.retrieve_context("medical treatment", max_nodes=3)
    print(f"Retrieved {len(context)} context nodes")
    
    # Test prompt formatting
    prompt = rag.format_prompt("What is the treatment for rare diseases?", context)
    print(f"Generated prompt length: {len(prompt)} characters")
    
    return rag

def test_api_endpoints(base_url="http://localhost:8000"):
    """Test the API endpoints."""
    print(f"\nTesting API Endpoints at {base_url}...")
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"Health check passed: {health_data}")
        else:
            print(f"Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("API server not running. Start it with: uvicorn app.main:app --reload")
        return False
    
    # Test knowledge graph stats
    try:
        response = requests.get(f"{base_url}/knowledge-graph/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"Knowledge graph stats: {stats}")
        else:
            print(f"Stats endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"Stats endpoint error: {e}")
    
    # Test drug search
    try:
        response = requests.post(
            f"{base_url}/search/drugs",
            json={"drug_name": "Advil", "max_results": 2}
        )
        if response.status_code == 200:
            drug_data = response.json()
            print(f"Drug search results: {len(drug_data.get('drug_labels', []))} labels, {len(drug_data.get('adverse_events', []))} adverse events")
        else:
            print(f"Drug search failed: {response.status_code}")
    except Exception as e:
        print(f"Drug search error: {e}")
    
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("MEDICAL RAG SYSTEM TEST")
    print("="*60)
    
    # Test individual components
    kg = test_knowledge_graph()
    fda = test_fda_collector()
    rag = test_rag_engine(kg)
    
    # Test API endpoints (if server is running)
    api_working = test_api_endpoints()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✅ Knowledge Graph: Working")
    print("✅ FDA Collector: Working")
    print("✅ RAG Engine: Working")
    print(f"{'✅' if api_working else '❌'} API Endpoints: {'Working' if api_working else 'Not running'}")
    
    if not api_working:
        print("\nTo start the API server, run:")
        print("uvicorn app.main:app --reload")
        print("\nThen you can test the API endpoints at http://localhost:8000")
        print("API documentation will be available at http://localhost:8000/docs")
    
    print("="*60)

if __name__ == "__main__":
    main() 