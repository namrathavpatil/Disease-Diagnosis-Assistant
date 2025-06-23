#!/usr/bin/env python3
"""
Script to initialize and populate the knowledge graph with sample data.
This script will add some diseases and articles to get started.
"""

import logging
import os
from app.core.knowledge_graph import KnowledgeGraph
from app.data.pubmed_collector import PubMedArticle
from app.data.orphanet_collector import OrphanetDisease
from app.data.fda_collector import FDACollector
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_knowledge_graph():
    """Initialize the knowledge graph with sample data."""
    
    # Initialize knowledge graph
    kg = KnowledgeGraph()
    logger.info("Initialized knowledge graph")
    
    # Add some diseases from Orphanet
    if settings.orphanet_api_key:
        logger.info("Adding diseases from Orphanet...")
        disease_ids = ["558", "166", "1001"]  # Marfan syndrome, Cystic fibrosis, Huntington disease
        
        for disease_id in disease_ids:
            try:
                disease = OrphanetDisease.from_api(disease_id)
                kg.add_disease(disease)
                logger.info(f"Added disease: {disease.name} (ID: {disease_id})")
            except Exception as e:
                logger.warning(f"Failed to add disease {disease_id}: {e}")
    
    # Add some articles from PubMed
    if settings.pubmed_api_key:
        logger.info("Adding articles from PubMed...")
        search_queries = [
            "rare diseases diagnosis",
            "genetic disorders treatment",
            "medical genetics research"
        ]
        
        for query in search_queries:
            try:
                articles = PubMedArticle.search(query, max_results=3)
                for article in articles:
                    kg.add_article(article)
                    logger.info(f"Added article: {article.title[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to add articles for query '{query}': {e}")
    
    # Add some drug information from FDA
    logger.info("Adding drug information from FDA...")
    fda_collector = FDACollector()
    
    drug_names = ["Advil", "Tylenol", "Aspirin"]
    for drug_name in drug_names:
        try:
            # Get drug labels
            drug_labels = fda_collector.search_drug_labels(
                f"openfda.brand_name:{drug_name}",
                limit=1
            )
            
            if not drug_labels.empty:
                drug_info = drug_labels.iloc[0]
                
                # Add drug node to knowledge graph
                drug_node_id = f"drug_{drug_name.lower()}"
                kg.graph.add_node(
                    drug_node_id,
                    type="drug",
                    name=drug_name,
                    metadata={
                        "description": drug_info.get("description", [""])[0] if drug_info.get("description") else "",
                        "indications": drug_info.get("indications_and_usage", [""])[0] if drug_info.get("indications_and_usage") else "",
                        "source": "FDA"
                    }
                )
                
                # Add adverse reactions as separate nodes
                adverse_reactions = drug_info.get("adverse_reactions", [])
                for i, reaction in enumerate(adverse_reactions[:3]):  # Limit to first 3 reactions
                    reaction_id = f"{drug_node_id}_reaction_{i}"
                    kg.graph.add_node(
                        reaction_id,
                        type="adverse_reaction",
                        name=reaction[:100] + "..." if len(reaction) > 100 else reaction,
                        metadata={"source": "FDA"}
                    )
                    kg.graph.add_edge(drug_node_id, reaction_id, type="causes_reaction")
                
                logger.info(f"Added drug: {drug_name}")
                
        except Exception as e:
            logger.warning(f"Failed to add drug {drug_name}: {e}")
    
    # Save the knowledge graph
    kg.save_graph("knowledge_graph.json")
    logger.info(f"Saved knowledge graph with {len(kg.graph.nodes)} nodes and {len(kg.graph.edges)} edges")
    
    # Print statistics
    print("\n" + "="*50)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*50)
    print(f"Total nodes: {len(kg.graph.nodes)}")
    print(f"Total edges: {len(kg.graph.edges)}")
    
    # Count by type
    node_types = {}
    for node in kg.graph.nodes():
        node_type = kg.graph.nodes[node].get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\nNodes by type:")
    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}")
    
    print(f"\nDiseases: {len(kg.disease_nodes)}")
    print(f"Articles: {len(kg.article_nodes)}")
    print("="*50)
    
    return kg

if __name__ == "__main__":
    initialize_knowledge_graph() 