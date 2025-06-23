#!/usr/bin/env python3
"""
Demonstration of different ways to build edges between entities from different collectors.
"""

import json
import os
from typing import Dict, List, Any
from app.data.pubmed_collector import PubMedArticle
from app.data.orphanet_collector import OrphanetDisease
from app.data.fda_collector import FDACollector

def build_edges_by_name_matching():
    """Build edges by matching entity names across collectors."""
    print("=== Building Edges by Name Matching ===")
    
    # Get data from collectors
    pubmed_articles = PubMedArticle.search("Cholera", max_results=2, api_key=os.getenv('PUBMED_API_KEY'))
    orphanet_diseases = OrphanetDisease.search_by_name("Cholera", api_key=os.getenv('ORPHANET_API_KEY'))
    
    edges = []
    
    # Link PubMed articles to Orphanet disease by name
    for article in pubmed_articles:
        for disease in orphanet_diseases:
            if "cholera" in article.title.lower():
                edges.append({
                    "source": f"pubmed_{article.pmid}",
                    "target": f"orphanet_{disease.disease_id}",
                    "type": "discusses_disease",
                    "confidence": 0.9,
                    "metadata": {
                        "source": "name_matching",
                        "article_title": article.title,
                        "disease_name": disease.name
                    }
                })
    
    print(f"Created {len(edges)} edges by name matching")
    return edges

def build_edges_by_cross_references():
    """Build edges using cross-references from Orphanet data."""
    print("\n=== Building Edges by Cross-References ===")
    
    orphanet_diseases = OrphanetDisease.search_by_name("Cholera", api_key=os.getenv('ORPHANET_API_KEY'))
    
    edges = []
    
    for disease in orphanet_diseases:
        # Extract cross-references from Orphanet metadata
        raw_data = disease.metadata.get("raw_data", {})
        data = raw_data.get("data", {})
        results = data.get("results", {})
        external_refs = results.get("ExternalReference", [])
        
        for ref in external_refs:
            source = ref.get("Source")
            reference = ref.get("Reference")
            relation = ref.get("DisorderMappingRelation")
            
            if source and reference:
                edges.append({
                    "source": f"orphanet_{disease.disease_id}",
                    "target": f"{source.lower()}_{reference}",
                    "type": "cross_references",
                    "confidence": 0.8,
                    "metadata": {
                        "source": "cross_reference",
                        "reference_source": source,
                        "reference_id": reference,
                        "relation_type": relation
                    }
                })
    
    print(f"Created {len(edges)} edges by cross-references")
    return edges

def build_edges_by_content_analysis():
    """Build edges by analyzing content and extracting related entities."""
    print("\n=== Building Edges by Content Analysis ===")
    
    pubmed_articles = PubMedArticle.search("Cholera", max_results=2, api_key=os.getenv('PUBMED_API_KEY'))
    orphanet_diseases = OrphanetDisease.search_by_name("Cholera", api_key=os.getenv('ORPHANET_API_KEY'))
    
    edges = []
    
    # Extract entities from Orphanet definition
    for disease in orphanet_diseases:
        raw_data = disease.metadata.get("raw_data", {})
        data = raw_data.get("data", {})
        results = data.get("results", {})
        summary_info = results.get("SummaryInformation", [])
        
        for summary in summary_info:
            definition = summary.get("Definition", "")
            
            # Extract key entities from definition
            entities = extract_entities_from_text(definition)
            
            for entity in entities:
                edges.append({
                    "source": f"orphanet_{disease.disease_id}",
                    "target": f"entity_{entity['text'].lower().replace(' ', '_')}",
                    "type": entity['relationship'],
                    "confidence": 0.7,
                    "metadata": {
                        "source": "content_analysis",
                        "entity_type": entity['type'],
                        "extracted_from": "orphanet_definition"
                    }
                })
    
    # Extract entities from PubMed titles/abstracts
    for article in pubmed_articles:
        if article.abstract:
            entities = extract_entities_from_text(article.abstract)
            
            for entity in entities:
                edges.append({
                    "source": f"pubmed_{article.pmid}",
                    "target": f"entity_{entity['text'].lower().replace(' ', '_')}",
                    "type": entity['relationship'],
                    "confidence": 0.6,
                    "metadata": {
                        "source": "content_analysis",
                        "entity_type": entity['type'],
                        "extracted_from": "pubmed_abstract"
                    }
                })
    
    print(f"Created {len(edges)} edges by content analysis")
    return edges

def extract_entities_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract entities and their relationships from text."""
    entities = []
    
    # Simple keyword extraction (in practice, you'd use NLP)
    keywords = {
        "Vibrio cholerae": {"type": "bacteria", "relationship": "caused_by"},
        "diarrhea": {"type": "symptom", "relationship": "has_symptom"},
        "dehydration": {"type": "symptom", "relationship": "has_symptom"},
        "infection": {"type": "condition", "relationship": "is_condition"},
        "treatment": {"type": "intervention", "relationship": "requires_treatment"},
        "vaccine": {"type": "intervention", "relationship": "prevented_by"},
        "antibiotics": {"type": "treatment", "relationship": "treated_with"}
    }
    
    text_lower = text.lower()
    for keyword, info in keywords.items():
        if keyword.lower() in text_lower:
            entities.append({
                "text": keyword,
                "type": info["type"],
                "relationship": info["relationship"]
            })
    
    return entities

def build_edges_by_semantic_similarity():
    """Build edges by finding semantically similar entities across collectors."""
    print("\n=== Building Edges by Semantic Similarity ===")
    
    # This would typically use embeddings or semantic similarity models
    # For demo purposes, we'll use simple keyword matching
    
    pubmed_articles = PubMedArticle.search("Cholera", max_results=2, api_key=os.getenv('PUBMED_API_KEY'))
    orphanet_diseases = OrphanetDisease.search_by_name("Cholera", api_key=os.getenv('ORPHANET_API_KEY'))
    
    edges = []
    
    # Extract MeSH terms from PubMed (if available)
    for article in pubmed_articles:
        mesh_terms = article.mesh_terms
        if mesh_terms:
            for term in mesh_terms:
                # Link MeSH terms to Orphanet cross-references
                for disease in orphanet_diseases:
                    raw_data = disease.metadata.get("raw_data", {})
                    data = raw_data.get("data", {})
                    results = data.get("results", {})
                    external_refs = results.get("ExternalReference", [])
                    
                    for ref in external_refs:
                        if ref.get("Source") == "MeSH" and ref.get("Reference"):
                            edges.append({
                                "source": f"pubmed_{article.pmid}",
                                "target": f"mesh_{ref['Reference']}",
                                "type": "uses_terminology",
                                "confidence": 0.8,
                                "metadata": {
                                    "source": "semantic_similarity",
                                    "mesh_term": term,
                                    "reference_id": ref["Reference"]
                                }
                            })
    
    print(f"Created {len(edges)} edges by semantic similarity")
    return edges

def build_edges_by_temporal_relationships():
    """Build edges based on temporal relationships (e.g., treatment timeline)."""
    print("\n=== Building Edges by Temporal Relationships ===")
    
    edges = []
    
    # Example: Link symptoms to treatments based on typical medical timeline
    timeline_entities = {
        "infection": {"stage": "onset", "next": ["symptoms"]},
        "symptoms": {"stage": "presentation", "next": ["diagnosis"]},
        "diagnosis": {"stage": "assessment", "next": ["treatment"]},
        "treatment": {"stage": "intervention", "next": ["recovery"]},
        "recovery": {"stage": "outcome", "next": []}
    }
    
    for entity, info in timeline_entities.items():
        for next_entity in info["next"]:
            edges.append({
                "source": f"entity_{entity}",
                "target": f"entity_{next_entity}",
                "type": "temporally_follows",
                "confidence": 0.9,
                "metadata": {
                    "source": "temporal_relationship",
                    "stage": info["stage"]
                }
            })
    
    print(f"Created {len(edges)} edges by temporal relationships")
    return edges

def main():
    """Demonstrate all edge building methods."""
    print("Building Edges Between Collector Entities")
    print("=" * 50)
    
    all_edges = []
    
    # Method 1: Name matching
    edges1 = build_edges_by_name_matching()
    all_edges.extend(edges1)
    
    # Method 2: Cross-references
    edges2 = build_edges_by_cross_references()
    all_edges.extend(edges2)
    
    # Method 3: Content analysis
    edges3 = build_edges_by_content_analysis()
    all_edges.extend(edges3)
    
    # Method 4: Semantic similarity
    edges4 = build_edges_by_semantic_similarity()
    all_edges.extend(edges4)
    
    # Method 5: Temporal relationships
    edges5 = build_edges_by_temporal_relationships()
    all_edges.extend(edges5)
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Total edges created: {len(all_edges)}")
    
    # Group by edge type
    edge_types = {}
    for edge in all_edges:
        edge_type = edge["type"]
        if edge_type not in edge_types:
            edge_types[edge_type] = 0
        edge_types[edge_type] += 1
    
    print("\nEdges by type:")
    for edge_type, count in edge_types.items():
        print(f"  {edge_type}: {count}")
    
    # Save to file
    with open("built_edges.json", "w") as f:
        json.dump(all_edges, f, indent=2)
    
    print(f"\nEdges saved to built_edges.json")
    
    # Show some example edges
    print(f"\n=== Example Edges ===")
    for i, edge in enumerate(all_edges[:5]):
        print(f"{i+1}. {edge['source']} --{edge['type']}--> {edge['target']} (confidence: {edge['confidence']})")

if __name__ == "__main__":
    main() 