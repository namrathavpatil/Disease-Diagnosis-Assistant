#!/usr/bin/env python3
"""
Comprehensive Knowledge Graph Builder using all collectors with FAISS.
Builds human-readable graphs by using information from one collector to inform searches in others.
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
from collections import defaultdict
import networkx as nx
from datetime import datetime

from app.data.pubmed_collector import PubMedArticle
from app.data.orphanet_collector import OrphanetDisease
from app.data.fda_collector import FDACollector

class ComprehensiveGraphBuilder:
    """Builds comprehensive knowledge graphs using all collectors with FAISS."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the graph builder with embedding model and FAISS index."""
        print(f"Initializing Comprehensive Graph Builder with {model_name}")
        self.model = SentenceTransformer(model_name)
        self.faiss_index = None
        self.entity_embeddings = {}
        self.entity_texts = {}
        self.entity_metadata = {}
        self.graph = nx.Graph()
        
        # Initialize collectors
        self.pubmed_collector = None  # Will be set if API key available
        self.orphanet_collector = None  # Will be set if API key available
        self.fda_collector = FDACollector()
        
        # Track entities by source
        self.entities_by_source = {
            'pubmed': [],
            'orphanet': [],
            'fda': []
        }
        
    def setup_collectors(self):
        """Setup collectors if API keys are available."""
        if os.getenv('PUBMED_API_KEY'):
            self.pubmed_collector = PubMedArticle
            print("✓ PubMed collector enabled")
        
        if os.getenv('ORPHANET_API_KEY'):
            self.orphanet_collector = OrphanetDisease
            print("✓ Orphanet collector enabled")
        
        print("✓ FDA collector enabled (no API key required)")
    
    def extract_entities_from_pubmed(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Extract entities from PubMed articles."""
        if not self.pubmed_collector:
            print("PubMed collector not available (no API key)")
            return []
        
        print(f"Searching PubMed for: {query}")
        entities = []
        
        try:
            articles = self.pubmed_collector.search(query, max_results=max_results, api_key=os.getenv('PUBMED_API_KEY'))
            
            for article in articles:
                # Create rich entity text
                entity_text = f"Article: {article.title}"
                if article.abstract:
                    entity_text += f" Abstract: {article.abstract}"
                if article.authors:
                    entity_text += f" Authors: {', '.join(article.authors[:3])}"
                if article.journal:
                    entity_text += f" Journal: {article.journal}"
                
                entity_id = f"pubmed_{article.pmid}"
                entities.append({
                    "id": entity_id,
                    "text": entity_text,
                    "type": "article",
                    "source": "pubmed",
                    "metadata": {
                        "title": article.title,
                        "abstract": article.abstract,
                        "authors": article.authors,
                        "journal": article.journal,
                        "publication_date": article.publication_date,
                        "pmid": article.pmid,
                        "mesh_terms": article.mesh_terms,
                        "chemicals": article.chemicals
                    }
                })
                
                # Extract key terms for cross-collector searches
                self._extract_search_terms(article, entity_id)
                
        except Exception as e:
            print(f"PubMed search failed: {e}")
        
        print(f"Extracted {len(entities)} PubMed entities")
        return entities
    
    def extract_entities_from_orphanet(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities from Orphanet."""
        if not self.orphanet_collector:
            print("Orphanet collector not available (no API key)")
            return []
        
        print(f"Searching Orphanet for: {query}")
        entities = []
        
        try:
            diseases = self.orphanet_collector.search_by_name(query, api_key=os.getenv('ORPHANET_API_KEY'))
            
            for disease in diseases:
                # Create rich entity text
                entity_text = f"Disease: {disease.name}"
                
                # Add definition
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
                    pheno_names = [p.get("name", "") for p in disease.phenotypes[:5]]
                    if any(pheno_names):
                        entity_text += f" Phenotypes: {', '.join(pheno_names)}"
                
                entity_id = f"orphanet_{disease.disease_id}"
                entities.append({
                    "id": entity_id,
                    "text": entity_text,
                    "type": "disease",
                    "source": "orphanet",
                    "metadata": {
                        "name": disease.name,
                        "definition": definition if 'definition' in locals() else "",
                        "prevalence": disease.prevalence,
                        "inheritance": disease.inheritance,
                        "phenotypes": disease.phenotypes,
                        "medical_specialties": disease.medical_specialties,
                        "icd10_codes": disease.icd10_codes,
                        "omim_ids": disease.omim_ids
                    }
                })
                
                # Extract search terms for cross-collector searches
                self._extract_search_terms_from_disease(disease, entity_id)
                
        except Exception as e:
            print(f"Orphanet search failed: {e}")
        
        print(f"Extracted {len(entities)} Orphanet entities")
        return entities
    
    def extract_entities_from_fda(self, search_terms: List[str], max_results: int = 3) -> List[Dict[str, Any]]:
        """Extract entities from FDA using search terms from other collectors."""
        print(f"Searching FDA with terms: {search_terms}")
        entities = []
        
        for term in search_terms:
            try:
                # Try different FDA search strategies
                search_queries = [
                    f"indications_and_usage:{term}",
                    f"description:{term}",
                    f"adverse_reactions:{term}",
                    f"active_ingredient:{term}"
                ]
                
                for search_query in search_queries:
                    try:
                        drug_labels = self.fda_collector.search_drug_labels(search_query, limit=max_results)
                        if not drug_labels.empty:
                            for _, row in drug_labels.iterrows():
                                drug_name = row['openfda'].get('brand_name', [term])[0] if row['openfda'].get('brand_name') else term
                                
                                # Create rich entity text
                                entity_text = f"Drug: {drug_name}"
                                
                                # Add description
                                description = row.get('description', [])
                                if description:
                                    entity_text += f" Description: {description[0][:300]}..."
                                
                                # Add indications
                                indications = row.get('indications_and_usage', [])
                                if indications:
                                    entity_text += f" Indications: {indications[0][:300]}..."
                                
                                # Add adverse reactions
                                adverse_reactions = row.get('adverse_reactions', [])
                                if adverse_reactions:
                                    entity_text += f" Adverse Reactions: {adverse_reactions[0][:200]}..."
                                
                                entity_id = f"fda_{drug_name.lower().replace(' ', '_').replace('-', '_')}"
                                
                                # Check if entity already exists
                                if not any(e["id"] == entity_id for e in entities):
                                    entities.append({
                                        "id": entity_id,
                                        "text": entity_text,
                                        "type": "drug",
                                        "source": "fda",
                                        "metadata": {
                                            "drug_name": drug_name,
                                            "description": description[0] if description else "",
                                            "indications": indications[0] if indications else "",
                                            "adverse_reactions": adverse_reactions[:3] if adverse_reactions else [],
                                            "active_ingredients": row['openfda'].get('substance_name', []),
                                            "search_term": term
                                        }
                                    })
                            
                            break  # Stop if we found results for this term
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"FDA search failed for term '{term}': {e}")
        
        print(f"Extracted {len(entities)} FDA entities")
        return entities
    
    def _extract_search_terms(self, article: PubMedArticle, entity_id: str):
        """Extract search terms from PubMed article for cross-collector searches."""
        search_terms = set()
        
        # Extract from title and abstract
        text = f"{article.title} {article.abstract or ''}"
        
        # Medical terms that might be useful for FDA searches
        medical_terms = [
            'antibiotic', 'antibiotics', 'treatment', 'therapy', 'medication', 'drug',
            'vaccine', 'vaccination', 'prevention', 'prophylaxis', 'antimicrobial',
            'bacterial', 'infection', 'infectious', 'pathogen', 'microorganism'
        ]
        
        text_lower = text.lower()
        for term in medical_terms:
            if term in text_lower:
                search_terms.add(term)
        
        # Store search terms for later use
        self.entity_metadata[entity_id] = self.entity_metadata.get(entity_id, {})
        self.entity_metadata[entity_id]['search_terms'] = list(search_terms)
    
    def _extract_search_terms_from_disease(self, disease: OrphanetDisease, entity_id: str):
        """Extract search terms from Orphanet disease for cross-collector searches."""
        search_terms = set()
        
        # Extract from disease name and definition
        text = f"{disease.name}"
        
        # Add definition if available
        raw_data = disease.metadata.get("raw_data", {})
        data = raw_data.get("data", {})
        results = data.get("results", {})
        summary_info = results.get("SummaryInformation", [])
        
        for summary in summary_info:
            definition = summary.get("Definition", "")
            if definition:
                text += f" {definition}"
        
        # Medical terms for FDA searches
        medical_terms = [
            'treatment', 'therapy', 'medication', 'drug', 'antibiotic', 'vaccine',
            'prevention', 'management', 'intervention', 'pharmacological'
        ]
        
        text_lower = text.lower()
        for term in medical_terms:
            if term in text_lower:
                search_terms.add(term)
        
        # Store search terms
        self.entity_metadata[entity_id] = self.entity_metadata.get(entity_id, {})
        self.entity_metadata[entity_id]['search_terms'] = list(search_terms)
    
    def build_faiss_index(self, entities: List[Dict[str, Any]]):
        """Build FAISS index for efficient similarity search."""
        print("Building FAISS index...")
        
        if not entities:
            print("No entities to index")
            return
        
        # Compute embeddings
        texts = [entity["text"] for entity in entities]
        embeddings = self.model.encode(texts)
        
        # Store entity information
        for i, entity in enumerate(entities):
            entity_id = entity["id"]
            self.entity_embeddings[entity_id] = embeddings[i]
            self.entity_texts[entity_id] = entity["text"]
            self.entity_metadata[entity_id] = entity["metadata"]
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(embeddings.astype('float32'))
        
        print(f"FAISS index built with {len(entities)} entities, dimension {dimension}")
    
    def find_similar_entities(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar entities using FAISS."""
        if self.faiss_index is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query_text])
        
        # Search FAISS index
        similarities, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        # Get entity IDs
        entity_ids = list(self.entity_embeddings.keys())
        results = []
        
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(entity_ids):
                entity_id = entity_ids[idx]
                results.append((entity_id, float(similarity)))
        
        return results
    
    def build_cross_collector_links(self, similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Build links between entities from different collectors."""
        print("Building cross-collector links...")
        
        entity_ids = list(self.entity_embeddings.keys())
        edges = []
        
        # Compute similarity matrix using FAISS
        embeddings_matrix = np.array([self.entity_embeddings[entity_id] for entity_id in entity_ids])
        
        # Use FAISS for efficient similarity computation
        self.faiss_index.reset()
        self.faiss_index.add(embeddings_matrix.astype('float32'))
        
        for i, source_id in enumerate(entity_ids):
            # Get similar entities for this source
            source_embedding = embeddings_matrix[i:i+1]
            similarities, indices = self.faiss_index.search(source_embedding.astype('float32'), len(entity_ids))
            
            for j, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(entity_ids) and i != idx:  # Avoid self-similarity
                    target_id = entity_ids[idx]
                    
                    if similarity >= similarity_threshold:
                        # Get source and target info
                        source_info = self._get_entity_info(source_id)
                        target_info = self._get_entity_info(target_id)
                        
                        # Only create edges between different sources
                        if source_info["source"] != target_info["source"]:
                            edge_type = self._determine_edge_type(source_info, target_info, similarity)
                            
                            edges.append({
                                "source": source_id,
                                "target": target_id,
                                "type": edge_type,
                                "confidence": float(similarity),
                                "metadata": {
                                    "source": "faiss_similarity",
                                    "similarity_score": float(similarity),
                                    "source_type": source_info["type"],
                                    "target_type": target_info["type"],
                                    "source_collector": source_info["source"],
                                    "target_collector": target_info["source"]
                                }
                            })
        
        print(f"Built {len(edges)} cross-collector links")
        return edges
    
    def _get_entity_info(self, entity_id: str) -> Dict[str, Any]:
        """Get entity information."""
        if entity_id.startswith("pubmed_"):
            return {"type": "article", "source": "pubmed"}
        elif entity_id.startswith("orphanet_"):
            return {"type": "disease", "source": "orphanet"}
        elif entity_id.startswith("fda_"):
            return {"type": "drug", "source": "fda"}
        else:
            return {"type": "unknown", "source": "unknown"}
    
    def _determine_edge_type(self, source_info: Dict, target_info: Dict, similarity: float) -> str:
        """Determine the type of edge based on entity types and similarity."""
        source_type = source_info["type"]
        target_type = target_info["type"]
        
        if source_type == "article" and target_type == "disease":
            return "discusses_disease"
        elif source_type == "disease" and target_type == "article":
            return "discussed_in_article"
        elif source_type == "drug" and target_type == "disease":
            return "treats_disease"
        elif source_type == "disease" and target_type == "drug":
            return "treated_by_drug"
        elif source_type == "article" and target_type == "drug":
            return "mentions_drug"
        elif source_type == "drug" and target_type == "article":
            return "mentioned_in_article"
        else:
            return "semantically_related"
    
    def build_comprehensive_graph(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Build a comprehensive knowledge graph using all collectors."""
        print(f"\n{'='*60}")
        print(f"Building Comprehensive Knowledge Graph for: {query}")
        print(f"{'='*60}")
        
        # Setup collectors
        self.setup_collectors()
        
        all_entities = []
        all_edges = []
        
        # Step 1: Extract entities from PubMed
        print(f"\n1. Extracting PubMed entities...")
        pubmed_entities = self.extract_entities_from_pubmed(query, max_results)
        all_entities.extend(pubmed_entities)
        self.entities_by_source['pubmed'] = pubmed_entities
        
        # Step 2: Extract entities from Orphanet
        print(f"\n2. Extracting Orphanet entities...")
        orphanet_entities = self.extract_entities_from_orphanet(query)
        all_entities.extend(orphanet_entities)
        self.entities_by_source['orphanet'] = orphanet_entities
        
        # Step 3: Extract search terms from PubMed and Orphanet
        print(f"\n3. Extracting search terms for FDA...")
        search_terms = set()
        
        # Collect search terms from all entities
        for entity in all_entities:
            entity_id = entity["id"]
            if entity_id in self.entity_metadata:
                terms = self.entity_metadata[entity_id].get('search_terms', [])
                search_terms.update(terms)
        
        # Add query itself as search term
        search_terms.add(query.lower())
        
        # Step 4: Extract FDA entities using search terms
        print(f"\n4. Extracting FDA entities using search terms: {list(search_terms)}")
        fda_entities = self.extract_entities_from_fda(list(search_terms), max_results)
        all_entities.extend(fda_entities)
        self.entities_by_source['fda'] = fda_entities
        
        # Step 5: Build FAISS index
        print(f"\n5. Building FAISS index...")
        self.build_faiss_index(all_entities)
        
        # Step 6: Build cross-collector links
        print(f"\n6. Building cross-collector links...")
        cross_links = self.build_cross_collector_links(similarity_threshold=0.6)
        all_edges.extend(cross_links)
        
        # Step 7: Build internal collector links
        print(f"\n7. Building internal collector links...")
        internal_links = self._build_internal_links()
        all_edges.extend(internal_links)
        
        # Step 8: Create human-readable graph
        print(f"\n8. Creating human-readable graph...")
        graph_data = self._create_human_readable_graph(all_entities, all_edges)
        
        # Step 9: Save results
        print(f"\n9. Saving results...")
        self._save_results(graph_data, all_entities, all_edges)
        
        return graph_data
    
    def _build_internal_links(self) -> List[Dict[str, Any]]:
        """Build links within each collector."""
        edges = []
        
        # PubMed internal links (articles discussing similar topics)
        pubmed_entities = self.entities_by_source['pubmed']
        for i, entity1 in enumerate(pubmed_entities):
            for j, entity2 in enumerate(pubmed_entities[i+1:], i+1):
                similar_entities = self.find_similar_entities(entity1["text"], top_k=len(pubmed_entities))
                for entity_id, similarity in similar_entities:
                    if entity_id == entity2["id"] and similarity > 0.7:
                        edges.append({
                            "source": entity1["id"],
                            "target": entity2["id"],
                            "type": "discusses_similar_topic",
                            "confidence": similarity,
                            "metadata": {
                                "source": "internal_similarity",
                                "collector": "pubmed"
                            }
                        })
        
        # Orphanet internal links (diseases with similar phenotypes)
        orphanet_entities = self.entities_by_source['orphanet']
        for entity in orphanet_entities:
            phenotypes = entity["metadata"].get("phenotypes", [])
            for phenotype in phenotypes[:3]:  # Limit to first 3 phenotypes
                pheno_name = phenotype.get("name", "")
                if pheno_name:
                    edges.append({
                        "source": entity["id"],
                        "target": f"phenotype_{pheno_name.lower().replace(' ', '_')}",
                        "type": "has_phenotype",
                        "confidence": 1.0,
                        "metadata": {
                            "source": "orphanet_phenotype",
                            "phenotype_name": pheno_name
                        }
                    })
        
        # FDA internal links (drugs with similar indications)
        fda_entities = self.entities_by_source['fda']
        for i, entity1 in enumerate(fda_entities):
            for j, entity2 in enumerate(fda_entities[i+1:], i+1):
                similar_entities = self.find_similar_entities(entity1["text"], top_k=len(fda_entities))
                for entity_id, similarity in similar_entities:
                    if entity_id == entity2["id"] and similarity > 0.7:
                        edges.append({
                            "source": entity1["id"],
                            "target": entity2["id"],
                            "type": "similar_indications",
                            "confidence": similarity,
                            "metadata": {
                                "source": "internal_similarity",
                                "collector": "fda"
                            }
                        })
        
        return edges
    
    def _create_human_readable_graph(self, entities: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a human-readable graph representation."""
        
        # Create nodes
        nodes = []
        for entity in entities:
            node = {
                "id": entity["id"],
                "type": entity["type"],
                "source": entity["source"],
                "name": self._extract_entity_name(entity),
                "description": self._extract_entity_description(entity),
                "metadata": entity["metadata"]
            }
            nodes.append(node)
        
        # Create edges
        graph_edges = []
        for edge in edges:
            graph_edge = {
                "source": edge["source"],
                "target": edge["target"],
                "type": edge["type"],
                "confidence": edge["confidence"],
                "description": self._create_edge_description(edge),
                "metadata": edge["metadata"]
            }
            graph_edges.append(graph_edge)
        
        return {
            "nodes": nodes,
            "edges": graph_edges,
            "summary": {
                "total_nodes": len(nodes),
                "total_edges": len(graph_edges),
                "nodes_by_source": {
                    source: len(entities) for source, entities in self.entities_by_source.items()
                },
                "edges_by_type": self._count_edges_by_type(graph_edges),
                "created_at": datetime.now().isoformat()
            }
        }
    
    def _extract_entity_name(self, entity: Dict[str, Any]) -> str:
        """Extract a human-readable name for an entity."""
        if entity["source"] == "pubmed":
            return entity["metadata"].get("title", entity["id"])
        elif entity["source"] == "orphanet":
            return entity["metadata"].get("name", entity["id"])
        elif entity["source"] == "fda":
            return entity["metadata"].get("drug_name", entity["id"])
        else:
            return entity["id"]
    
    def _extract_entity_description(self, entity: Dict[str, Any]) -> str:
        """Extract a human-readable description for an entity."""
        if entity["source"] == "pubmed":
            abstract = entity["metadata"].get("abstract", "")
            return abstract[:200] + "..." if len(abstract) > 200 else abstract
        elif entity["source"] == "orphanet":
            return entity["metadata"].get("definition", "")
        elif entity["source"] == "fda":
            description = entity["metadata"].get("description", "")
            return description[:200] + "..." if len(description) > 200 else description
        else:
            return ""
    
    def _create_edge_description(self, edge: Dict[str, Any]) -> str:
        """Create a human-readable description for an edge."""
        edge_type = edge["type"]
        confidence = edge["confidence"]
        
        descriptions = {
            "discusses_disease": f"Article discusses the disease (confidence: {confidence:.2f})",
            "discussed_in_article": f"Disease is discussed in this article (confidence: {confidence:.2f})",
            "treats_disease": f"Drug treats the disease (confidence: {confidence:.2f})",
            "treated_by_drug": f"Disease is treated by this drug (confidence: {confidence:.2f})",
            "mentions_drug": f"Article mentions this drug (confidence: {confidence:.2f})",
            "mentioned_in_article": f"Drug is mentioned in this article (confidence: {confidence:.2f})",
            "semantically_related": f"Semantically related (confidence: {confidence:.2f})",
            "has_phenotype": "Disease has this phenotype",
            "discusses_similar_topic": f"Articles discuss similar topics (confidence: {confidence:.2f})",
            "similar_indications": f"Drugs have similar indications (confidence: {confidence:.2f})"
        }
        
        return descriptions.get(edge_type, f"Related (confidence: {confidence:.2f})")
    
    def _count_edges_by_type(self, edges: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count edges by type."""
        counts = defaultdict(int)
        for edge in edges:
            counts[edge["type"]] += 1
        return dict(counts)
    
    def _save_results(self, graph_data: Dict[str, Any], entities: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
        """Save results to files."""
        
        # Save comprehensive graph
        with open("comprehensive_graph.json", "w") as f:
            json.dump(graph_data, f, indent=2)
        
        # Save raw entities
        with open("comprehensive_entities.json", "w") as f:
            json.dump(entities, f, indent=2)
        
        # Save raw edges
        with open("comprehensive_edges.json", "w") as f:
            json.dump(edges, f, indent=2)
        
        # Save human-readable summary
        summary = {
            "query": "Cholera",  # This should be passed as parameter
            "timestamp": datetime.now().isoformat(),
            "summary": graph_data["summary"],
            "entities_by_source": {
                source: [
                    {
                        "id": entity["id"],
                        "name": self._extract_entity_name(entity),
                        "type": entity["type"]
                    }
                    for entity in entities
                ]
                for source, entities in self.entities_by_source.items()
            },
            "top_edges": [
                {
                    "source": edge["source"],
                    "target": edge["target"],
                    "type": edge["type"],
                    "confidence": edge["confidence"],
                    "description": self._create_edge_description(edge)
                }
                for edge in sorted(edges, key=lambda x: x["confidence"], reverse=True)[:10]
            ]
        }
        
        with open("comprehensive_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("Results saved to:")
        print("  - comprehensive_graph.json (full graph)")
        print("  - comprehensive_entities.json (raw entities)")
        print("  - comprehensive_edges.json (raw edges)")
        print("  - comprehensive_summary.json (human-readable summary)")

def main():
    """Main function to demonstrate comprehensive graph building."""
    print("Comprehensive Knowledge Graph Builder")
    print("Using all collectors with FAISS and cross-collector information flow")
    print("=" * 80)
    
    # Initialize builder
    builder = ComprehensiveGraphBuilder()
    
    # Build comprehensive graph
    query = "Cholera"
    graph_data = builder.build_comprehensive_graph(query, max_results=3)
    
    # Display results
    print(f"\n{'='*60}")
    print("BUILD COMPLETE!")
    print(f"{'='*60}")
    
    summary = graph_data["summary"]
    print(f"\n📊 Graph Summary:")
    print(f"   Total Nodes: {summary['total_nodes']}")
    print(f"   Total Edges: {summary['total_edges']}")
    print(f"   Nodes by Source:")
    for source, count in summary['nodes_by_source'].items():
        print(f"     {source.capitalize()}: {count}")
    print(f"   Edges by Type:")
    for edge_type, count in summary['edges_by_type'].items():
        print(f"     {edge_type}: {count}")
    
    print(f"\n🔗 Top Connections:")
    top_edges = sorted(graph_data["edges"], key=lambda x: x["confidence"], reverse=True)[:5]
    for i, edge in enumerate(top_edges, 1):
        source_name = next((n["name"] for n in graph_data["nodes"] if n["id"] == edge["source"]), edge["source"])
        target_name = next((n["name"] for n in graph_data["nodes"] if n["id"] == edge["target"]), edge["target"])
        print(f"   {i}. {source_name} → {target_name}")
        print(f"      Type: {edge['type']} (confidence: {edge['confidence']:.3f})")
        print(f"      {edge['description']}")
    
    print(f"\n📁 Files saved:")
    print(f"   - comprehensive_graph.json")
    print(f"   - comprehensive_entities.json") 
    print(f"   - comprehensive_edges.json")
    print(f"   - comprehensive_summary.json")

if __name__ == "__main__":
    main() 