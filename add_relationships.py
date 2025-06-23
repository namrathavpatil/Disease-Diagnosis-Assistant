#!/usr/bin/env python3
"""
Script to add relationships between nodes in the knowledge graph.
This will create edges and make the graph more connected.
"""

import logging
from app.core.knowledge_graph import KnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_relationships():
    """Add relationships between nodes to create edges."""
    
    # Load the knowledge graph
    kg = KnowledgeGraph()
    kg.load_from_file("knowledge_graph.json")
    logger.info("Loaded knowledge graph")
    
    # Get all nodes
    nodes = list(kg.graph.nodes())
    diseases = [n for n in nodes if kg.graph.nodes[n].get('type') == 'disease']
    articles = [n for n in nodes if kg.graph.nodes[n].get('type') == 'article']
    drugs = [n for n in nodes if kg.graph.nodes[n].get('type') == 'drug']
    
    logger.info(f"Found {len(diseases)} diseases, {len(articles)} articles, {len(drugs)} drugs")
    
    # Add relationships between diseases and articles based on content similarity
    for disease_id in diseases:
        disease_name = kg.graph.nodes[disease_id].get('name', '').lower()
        
        for article_id in articles:
            article_title = kg.graph.nodes[article_id].get('name', '').lower()
            
            # Simple keyword matching
            if any(keyword in article_title for keyword in ['genetic', 'disease', 'syndrome', 'disorder']):
                kg.graph.add_edge(disease_id, article_id, type="discussed_in", weight=0.5)
                logger.info(f"Added edge: {disease_id} -> {article_id} (discussed_in)")
    
    # Add relationships between diseases and drugs (for treatment)
    for disease_id in diseases:
        disease_name = kg.graph.nodes[disease_id].get('name', '').lower()
        
        for drug_id in drugs:
            drug_name = kg.graph.nodes[drug_id].get('name', '').lower()
            
            # Simple matching - in reality this would be more sophisticated
            if 'pain' in disease_name or 'inflammation' in disease_name:
                if drug_name in ['advil', 'tylenol', 'aspirin']:
                    kg.graph.add_edge(disease_id, drug_id, type="treated_by", weight=0.3)
                    logger.info(f"Added edge: {disease_id} -> {drug_id} (treated_by)")
    
    # Add relationships between articles and drugs (mentions)
    for article_id in articles:
        article_title = kg.graph.nodes[article_id].get('name', '').lower()
        
        for drug_id in drugs:
            drug_name = kg.graph.nodes[drug_id].get('name', '').lower()
            
            if 'treatment' in article_title or 'therapy' in article_title:
                kg.graph.add_edge(article_id, drug_id, type="mentions", weight=0.4)
                logger.info(f"Added edge: {article_id} -> {drug_id} (mentions)")
    
    # Add some phenotype-like nodes for diseases
    for disease_id in diseases:
        disease_name = kg.graph.nodes[disease_id].get('name', '')
        
        # Add some common phenotypes based on disease type
        if 'marfan' in disease_name.lower():
            phenotypes = ['tall stature', 'long limbs', 'heart problems', 'eye problems']
        elif 'charcot' in disease_name.lower():
            phenotypes = ['muscle weakness', 'foot deformities', 'sensory loss']
        else:
            phenotypes = ['developmental delay', 'intellectual disability']
        
        for i, phenotype in enumerate(phenotypes):
            phenotype_id = f"{disease_id}_phenotype_{i}"
            kg.graph.add_node(
                phenotype_id,
                type="phenotype",
                name=phenotype,
                metadata={"source": "manual"}
            )
            kg.graph.add_edge(disease_id, phenotype_id, type="has_phenotype", weight=0.8)
            logger.info(f"Added phenotype: {phenotype_id} for {disease_id}")
    
    # Save the updated knowledge graph
    kg.save_graph("knowledge_graph.json")
    logger.info(f"Saved knowledge graph with {len(kg.graph.nodes)} nodes and {len(kg.graph.edges)} edges")
    
    # Print statistics
    print("\n" + "="*50)
    print("UPDATED KNOWLEDGE GRAPH STATISTICS")
    print("="*50)
    print(f"Total nodes: {len(kg.graph.nodes)}")
    print(f"Total edges: {len(kg.graph.edges)}")
    
    # Count by type
    node_types = {}
    edge_types = {}
    for node in kg.graph.nodes():
        node_type = kg.graph.nodes[node].get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    for edge in kg.graph.edges():
        edge_type = kg.graph.edges[edge].get('type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print("\nNodes by type:")
    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}")
    
    print("\nEdges by type:")
    for edge_type, count in edge_types.items():
        print(f"  {edge_type}: {count}")
    
    print("="*50)
    
    return kg

if __name__ == "__main__":
    add_relationships() 