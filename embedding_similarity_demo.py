#!/usr/bin/env python3
"""
Demonstration of embedding-based similarity for linking entities across collectors.
"""

import json
import os
import numpy as np
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.data.pubmed_collector import PubMedArticle
from app.data.orphanet_collector import OrphanetDisease
from app.data.fda_collector import FDACollector

class EmbeddingSimilarityLinker:
    """Class to link entities across collectors using embedding similarity."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with a sentence transformer model."""
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.entity_embeddings = {}
        self.entity_texts = {}
        
    def extract_entity_texts(self, query: str = "Cholera"):
        """Extract entity texts from all collectors."""
        print(f"Extracting entity texts for query: {query}")
        
        entities = []
        
        # --- PubMed Entities ---
        try:
            pubmed_articles = PubMedArticle.search(query, max_results=3, api_key=os.getenv('PUBMED_API_KEY'))
            for article in pubmed_articles:
                # Create entity text from title and abstract
                entity_text = f"Article: {article.title}"
                if article.abstract:
                    entity_text += f" Abstract: {article.abstract[:200]}..."
                
                entities.append({
                    "id": f"pubmed_{article.pmid}",
                    "text": entity_text,
                    "type": "article",
                    "source": "pubmed",
                    "metadata": {
                        "title": article.title,
                        "journal": article.journal,
                        "authors": article.authors[:3] if article.authors else []
                    }
                })
        except Exception as e:
            print(f"PubMed extraction failed: {e}")
        
        # --- Orphanet Entities ---
        try:
            orphanet_diseases = OrphanetDisease.search_by_name(query, api_key=os.getenv('ORPHANET_API_KEY'))
            for disease in orphanet_diseases:
                # Create entity text from disease info
                entity_text = f"Disease: {disease.name}"
                
                # Add definition if available
                raw_data = disease.metadata.get("raw_data", {})
                data = raw_data.get("data", {})
                results = data.get("results", {})
                summary_info = results.get("SummaryInformation", [])
                
                for summary in summary_info:
                    definition = summary.get("Definition", "")
                    if definition:
                        entity_text += f" Definition: {definition}"
                
                # Add phenotypes
                if disease.phenotypes:
                    pheno_names = [p.get("name", "") for p in disease.phenotypes[:3]]
                    if any(pheno_names):
                        entity_text += f" Phenotypes: {', '.join(pheno_names)}"
                
                entities.append({
                    "id": f"orphanet_{disease.disease_id}",
                    "text": entity_text,
                    "type": "disease",
                    "source": "orphanet",
                    "metadata": {
                        "name": disease.name,
                        "prevalence": disease.prevalence,
                        "phenotypes_count": len(disease.phenotypes)
                    }
                })
                
                # Add individual phenotypes as separate entities
                for i, phenotype in enumerate(disease.phenotypes[:5]):  # Limit to first 5
                    pheno_name = phenotype.get("name", "")
                    if pheno_name:
                        entities.append({
                            "id": f"phenotype_{disease.disease_id}_{i}",
                            "text": f"Phenotype: {pheno_name}",
                            "type": "phenotype",
                            "source": "orphanet",
                            "metadata": {
                                "parent_disease": disease.name,
                                "phenotype_name": pheno_name
                            }
                        })
        except Exception as e:
            print(f"Orphanet extraction failed: {e}")
        
        # --- FDA Entities ---
        try:
            fda_collector = FDACollector()
            
            # Try different FDA searches
            search_queries = [
                f"indications_and_usage:{query}",
                f"description:{query}",
                f"adverse_reactions:{query}"
            ]
            
            for search_query in search_queries:
                try:
                    drug_labels = fda_collector.search_drug_labels(search_query, limit=2)
                    if not drug_labels.empty:
                        for _, row in drug_labels.iterrows():
                            drug_name = row['openfda'].get('brand_name', [query])[0] if row['openfda'].get('brand_name') else query
                            
                            # Create entity text from drug info
                            entity_text = f"Drug: {drug_name}"
                            
                            # Add description
                            description = row.get('description', [])
                            if description:
                                entity_text += f" Description: {description[0][:200]}..."
                            
                            # Add indications
                            indications = row.get('indications_and_usage', [])
                            if indications:
                                entity_text += f" Indications: {indications[0][:200]}..."
                            
                            entities.append({
                                "id": f"fda_{drug_name.lower().replace(' ', '_')}",
                                "text": entity_text,
                                "type": "drug",
                                "source": "fda",
                                "metadata": {
                                    "drug_name": drug_name,
                                    "has_description": bool(description),
                                    "has_indications": bool(indications)
                                }
                            })
                        break  # Stop if we found results
                except Exception as e:
                    continue
        except Exception as e:
            print(f"FDA extraction failed: {e}")
        
        print(f"Extracted {len(entities)} entities from all collectors")
        return entities
    
    def compute_embeddings(self, entities: List[Dict[str, Any]]):
        """Compute embeddings for all entities."""
        print("Computing embeddings for entities...")
        
        for entity in entities:
            self.entity_texts[entity["id"]] = entity["text"]
            self.entity_embeddings[entity["id"]] = self.model.encode(entity["text"])
        
        print(f"Computed embeddings for {len(self.entity_embeddings)} entities")
    
    def find_similar_entities(self, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find similar entities across different collectors."""
        print(f"Finding similar entities with threshold: {similarity_threshold}")
        
        entity_ids = list(self.entity_embeddings.keys())
        edges = []
        
        # Compute similarity matrix
        embeddings_matrix = np.array([self.entity_embeddings[entity_id] for entity_id in entity_ids])
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Find pairs above threshold
        for i, source_id in enumerate(entity_ids):
            for j, target_id in enumerate(entity_ids):
                if i < j:  # Avoid duplicate pairs and self-similarity
                    similarity = similarity_matrix[i][j]
                    
                    if similarity >= similarity_threshold:
                        # Get entity info
                        source_entity = self._get_entity_info(source_id)
                        target_entity = self._get_entity_info(target_id)
                        
                        # Only create edges between different sources
                        if source_entity["source"] != target_entity["source"]:
                            edges.append({
                                "source": source_id,
                                "target": target_id,
                                "type": "semantically_similar",
                                "confidence": float(similarity),
                                "metadata": {
                                    "source": "embedding_similarity",
                                    "similarity_score": float(similarity),
                                    "source_type": source_entity["type"],
                                    "target_type": target_entity["type"],
                                    "source_collector": source_entity["source"],
                                    "target_collector": target_entity["source"]
                                }
                            })
        
        print(f"Found {len(edges)} similar entity pairs")
        return edges
    
    def _get_entity_info(self, entity_id: str) -> Dict[str, Any]:
        """Get entity information from the original entities list."""
        # This would need to be implemented based on how you store entity info
        # For now, return basic info
        if entity_id.startswith("pubmed_"):
            return {"type": "article", "source": "pubmed"}
        elif entity_id.startswith("orphanet_"):
            return {"type": "disease", "source": "orphanet"}
        elif entity_id.startswith("phenotype_"):
            return {"type": "phenotype", "source": "orphanet"}
        elif entity_id.startswith("fda_"):
            return {"type": "drug", "source": "fda"}
        else:
            return {"type": "unknown", "source": "unknown"}
    
    def find_most_similar_entities(self, entity_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the most similar entities to a given entity."""
        if entity_id not in self.entity_embeddings:
            return []
        
        entity_embedding = self.entity_embeddings[entity_id]
        similarities = []
        
        for other_id, other_embedding in self.entity_embeddings.items():
            if other_id != entity_id:
                similarity = cosine_similarity([entity_embedding], [other_embedding])[0][0]
                similarities.append((other_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

def main():
    """Demonstrate embedding-based similarity linking."""
    print("Embedding-Based Similarity for Entity Linking")
    print("=" * 50)
    
    # Initialize the linker
    linker = EmbeddingSimilarityLinker()
    
    # Extract entities from all collectors
    entities = linker.extract_entity_texts("Cholera")
    
    # Show extracted entities
    print(f"\n=== Extracted Entities ===")
    for entity in entities:
        print(f"{entity['id']} ({entity['type']} from {entity['source']}): {entity['text'][:100]}...")
    
    # Compute embeddings
    linker.compute_embeddings(entities)
    
    # Find similar entities
    edges = linker.find_similar_entities(similarity_threshold=0.6)
    
    # Show results
    print(f"\n=== Similarity Results ===")
    print(f"Total edges found: {len(edges)}")
    
    # Group by edge type
    edge_types = {}
    for edge in edges:
        edge_type = edge["type"]
        if edge_type not in edge_types:
            edge_types[edge_type] = 0
        edge_types[edge_type] += 1
    
    print("\nEdges by type:")
    for edge_type, count in edge_types.items():
        print(f"  {edge_type}: {count}")
    
    # Show top similarity scores
    print(f"\n=== Top Similarity Scores ===")
    edges_sorted = sorted(edges, key=lambda x: x["confidence"], reverse=True)
    for i, edge in enumerate(edges_sorted[:10]):
        print(f"{i+1}. {edge['source']} ↔ {edge['target']} (similarity: {edge['confidence']:.3f})")
        print(f"   Source: {edge['metadata']['source_collector']} → Target: {edge['metadata']['target_collector']}")
    
    # Find most similar entities for a specific entity
    if entities:
        sample_entity = entities[0]["id"]
        print(f"\n=== Most Similar to {sample_entity} ===")
        similar_entities = linker.find_most_similar_entities(sample_entity, top_k=3)
        for entity_id, similarity in similar_entities:
            print(f"  {entity_id}: {similarity:.3f}")
    
    # Save results
    with open("embedding_similarity_edges.json", "w") as f:
        json.dump(edges, f, indent=2)
    
    print(f"\nResults saved to embedding_similarity_edges.json")
    
    # Show entity texts for top matches
    print(f"\n=== Entity Text Comparison (Top Match) ===")
    if edges_sorted:
        top_edge = edges_sorted[0]
        source_text = linker.entity_texts.get(top_edge["source"], "N/A")
        target_text = linker.entity_texts.get(top_edge["target"], "N/A")
        
        print(f"Source ({top_edge['source']}):")
        print(f"  {source_text[:200]}...")
        print(f"\nTarget ({top_edge['target']}):")
        print(f"  {target_text[:200]}...")
        print(f"\nSimilarity: {top_edge['confidence']:.3f}")

if __name__ == "__main__":
    main() 