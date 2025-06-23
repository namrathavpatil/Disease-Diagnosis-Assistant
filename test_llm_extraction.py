#!/usr/bin/env python3
"""
Test script for DeepSeek LLM entity extraction functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.entity_extractor_llm import LLMEntityExtractor

def test_llm_entity_extraction():
    """Test DeepSeek LLM entity extraction."""
    print("Testing DeepSeek LLM Entity Extraction...")
    
    # Initialize LLM extractor
    extractor = LLMEntityExtractor()
    
    # Test texts
    test_texts = [
        "Diabetes causes high blood sugar. Metformin is used to treat diabetes and helps control blood glucose levels.",
        "Aspirin treats pain and reduces fever. It is used for heart disease prevention.",
        "Marfan syndrome is a genetic disorder that affects connective tissue. It can cause heart problems and vision issues.",
        "Ibuprofen relieves pain and inflammation. It is commonly used for headaches and arthritis.",
        "Hypertension can lead to heart disease and stroke. Lisinopril is prescribed to control blood pressure."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Text: {text}")
        
        # Extract entities and relationships
        result = extractor.extract_entities_with_relationships(text)
        
        print(f"Extracted {len(result['entities'])} entities:")
        for entity in result['entities']:
            print(f"  - {entity['type']}: '{entity['text']}' (confidence: {entity['confidence']:.2f})")
        
        print(f"Built {len(result['relationships'])} relationships:")
        for rel in result['relationships']:
            print(f"  - {rel['source']} --{rel['type']}--> {rel['target']} (confidence: {rel['confidence']:.2f})")

def test_simple_entity_extraction():
    """Test simple entity extraction without relationships."""
    print("\n\nTesting Simple Entity Extraction...")
    
    extractor = LLMEntityExtractor()
    
    test_text = "What are the side effects of Metformin in patients with diabetes and hypertension?"
    
    print(f"Text: {test_text}")
    
    entities = extractor.extract_entities(test_text)
    
    print(f"Extracted {len(entities)} entities:")
    for entity in entities:
        print(f"  - {entity['type']}: '{entity['text']}' (confidence: {entity['confidence']:.2f})")

if __name__ == "__main__":
    print("DeepSeek LLM Entity Extraction Test")
    print("=" * 50)
    
    try:
        test_llm_entity_extraction()
        test_simple_entity_extraction()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 