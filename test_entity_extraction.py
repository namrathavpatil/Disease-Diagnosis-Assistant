#!/usr/bin/env python3
"""
Test script for entity extraction and knowledge graph building functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.entity_extractor import MedicalEntityExtractor
from app.core.knowledge_graph import KnowledgeGraph
from app.data.orphanet_collector import OrphanetDisease

def test_entity_extraction():
    """Test entity extraction from medical text."""
    print("Testing Entity Extraction...")
    
    # Initialize entity extractor
    extractor = MedicalEntityExtractor()
    
    # Test texts
    test_texts = [
        "Diabetes causes high blood sugar. Metformin is used to treat diabetes and helps control blood glucose levels.",
        "Aspirin treats pain and reduces fever. It is used for heart disease prevention.",
        "Marfan syndrome is a genetic disorder that affects connective tissue. It can cause heart problems and vision issues.",
        "Ibuprofen relieves pain and inflammation. It is commonly used for headaches and arthritis."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Text: {text}")
        
        # Extract entities
        entities = extractor.extract_entities(text)
        print(f"Extracted {len(entities)} entities:")
        
        for entity in entities:
            print(f"  - {entity.type}: '{entity.text}' (confidence: {entity.confidence:.2f})")
        
        # Build relationships
        relationships = extractor.build_relationships(entities, text)
        print(f"Built {len(relationships)} relationships:")
        
        for rel in relationships:
            print(f"  - {rel['source']} --{rel['type']}--> {rel['target']} (confidence: {rel['confidence']:.2f})")

def test_knowledge_graph_building():
    """Test knowledge graph building from medical text."""
    print("\n\nTesting Knowledge Graph Building...")
    
    # Initialize components
    extractor = MedicalEntityExtractor()
    knowledge_graph = KnowledgeGraph()
    
    # Test text
    test_text = "Diabetes causes high blood sugar. Metformin is used to treat diabetes and helps control blood glucose levels. Pain and fatigue are common symptoms of diabetes."
    
    print(f"Processing text: {test_text}")
    
    # Process the text to build knowledge graph
    result = knowledge_graph.process_medical_question(test_text, extractor)
    
    print(f"Added {result['total_entities']} entities and {result['total_relationships']} relationships")
    
    # Display entities
    print("\nEntities added:")
    for entity in result['entities']:
        print(f"  - {entity['type']}: {entity['text']}")
    
    # Display relationships
    print("\nRelationships built:")
    for rel in result['relationships']:
        print(f"  - {rel['source']} --{rel['type']}--> {rel['target']}")
    
    # Get graph stats
    nodes = list(knowledge_graph.graph.nodes())
    edges = list(knowledge_graph.graph.edges())
    
    print(f"\nKnowledge Graph Stats:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Total edges: {len(edges)}")
    
    # Display node types
    node_types = {}
    for node in nodes:
        node_type = knowledge_graph.graph.nodes[node].get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"  Node types: {node_types}")

def test_entity_network():
    """Test getting entity network."""
    print("\n\nTesting Entity Network...")
    
    # Initialize components
    extractor = MedicalEntityExtractor()
    knowledge_graph = KnowledgeGraph()
    
    # Build graph from text
    test_text = "Diabetes causes high blood sugar. Metformin treats diabetes. Pain is a symptom of diabetes."
    knowledge_graph.process_medical_question(test_text, extractor)
    
    # Get network for diabetes
    network = knowledge_graph.get_entity_network("diabetes", "disease")
    
    print(f"Network for 'diabetes':")
    print(f"  Entity: {network['entity']}")
    print(f"  Neighbors: {len(network['neighbors'])}")
    for neighbor in network['neighbors']:
        print(f"    - {neighbor['name']} ({neighbor['type']})")
    
    print(f"  Relationships: {len(network['relationships'])}")
    for rel in network['relationships']:
        print(f"    - {rel['source']} --{rel['type']}--> {rel['target']}")

if __name__ == "__main__":
    print("Medical Entity Extraction and Knowledge Graph Building Test")
    print("=" * 60)
    
    try:
        test_entity_extraction()
        test_knowledge_graph_building()
        test_entity_network()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 