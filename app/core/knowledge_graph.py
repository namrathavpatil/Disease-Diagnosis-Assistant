import json
import logging
import os
from typing import List, Dict, Any, Optional, Set
import networkx as nx
from pydantic import BaseModel
from app.data.orphanet_collector import OrphanetDisease
from sentence_transformers import SentenceTransformer
import numpy as np
from app.data.pubmed_collector import PubMedArticle

logger = logging.getLogger(__name__)

class KnowledgeGraphNode(BaseModel):
    """Model for knowledge graph nodes."""
    id: str
    type: str  # 'disease', 'phenotype', 'gene', etc.
    name: str
    metadata: Dict[str, Any]

class KnowledgeGraphEdge(BaseModel):
    """Model for knowledge graph edges."""
    source: str
    target: str
    type: str  # 'has_phenotype', 'caused_by', etc.
    weight: float = 1.0
    metadata: Dict[str, Any] = {}

class KnowledgeGraph:
    """A knowledge graph that integrates Orphanet and PubMed data."""

    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.graph = nx.DiGraph()
        self.disease_nodes: Dict[str, Dict[str, Any]] = {}
        self.article_nodes: Dict[str, Dict[str, Any]] = {}
        self.mesh_term_nodes: Dict[str, Dict[str, Any]] = {}
        self.chemical_nodes: Dict[str, Dict[str, Any]] = {}
        # Initialize sentence transformer for semantic search
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.node_embeddings = {}
        self._compute_node_embeddings()

    def _compute_node_embeddings(self):
        """Compute embeddings for all nodes in the graph."""
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            # Combine node type, name and metadata into a single text
            text = f"{node_data['type']} {node_data['name']}"
            if 'metadata' in node_data:
                for key, value in node_data['metadata'].items():
                    if isinstance(value, str):
                        text += f" {value}"
                    elif isinstance(value, list):
                        text += f" {' '.join(str(v) for v in value)}"
            # Compute embedding
            self.node_embeddings[node_id] = self.model.encode(text)

    def search_nodes(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for nodes semantically similar to the query."""
        if not self.graph.nodes():
            return []

        # Compute query embedding
        query_embedding = self.model.encode(query)

        # Compute similarities
        similarities = {}
        for node_id, node_embedding in self.node_embeddings.items():
            similarity = np.dot(query_embedding, node_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding)
            )
            similarities[node_id] = similarity

        # Get top k nodes
        top_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Format results
        results = []
        for node_id, score in top_nodes:
            node_data = self.graph.nodes[node_id]
            results.append({
                'id': node_id,
                'type': node_data['type'],
                'name': node_data['name'],
                'metadata': node_data.get('metadata', {}),
                'similarity_score': float(score)
            })

        return results

    def load_from_file(self, file_path):
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)
            
            # Clear existing graph
            self.graph.clear()
            
            # Add nodes
            for node in data.get('nodes', []):
                self.graph.add_node(
                    node['id'],
                    type=node['type'],
                    name=node['name'],
                    metadata=node.get('metadata', {})
                )
            
            # Add edges
            for edge in data.get('links', []):
                self.graph.add_edge(
                    edge['source'],
                    edge['target'],
                    type=edge['type'],
                    weight=edge.get('weight', 1.0),
                    metadata=edge.get('metadata', {})
                )
            
            logging.info(f"Loaded {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges from {file_path}")
            
            # Recompute embeddings after loading
            self._compute_node_embeddings()
    
    def add_disease(self, disease: OrphanetDisease) -> None:
        """
        Add a disease node and its relationships to the graph.
        
        Args:
            disease: OrphanetDisease instance to add
        """
        node = disease.to_dict()
        node['type'] = 'disease'
        # Ensure 'name' exists
        if 'name' not in node or not node['name']:
            node['name'] = node.get('disease_id', 'Unknown Disease')
        self.graph.add_node(node['disease_id'], **node)
        self.disease_nodes[node['disease_id']] = node

        # Add phenotype relationships
        for phenotype in disease.phenotypes:
            phenotype_id = f"phenotype_{phenotype.get('id', '')}"
            self.graph.add_node(phenotype_id, type="phenotype", **phenotype)
            self.graph.add_edge(node['disease_id'], phenotype_id, relationship="has_phenotype")

        # Add medical specialty relationships
        for specialty in disease.medical_specialties:
            specialty_id = f"specialty_{specialty.lower().replace(' ', '_')}"
            self.graph.add_node(specialty_id, type="specialty", name=specialty)
            self.graph.add_edge(node['disease_id'], specialty_id, relationship="treated_by")

        # Recompute embeddings after adding new nodes
        self._compute_node_embeddings()
    
    def get_disease_phenotypes(self, orpha_code: str) -> List[Dict[str, Any]]:
        """Get all phenotypes associated with a disease."""
        phenotypes = []
        for _, phenotype_id in self.graph.edges(orpha_code):
            if self.graph.nodes[phenotype_id]["type"] == "phenotype":
                phenotypes.append({
                    "id": phenotype_id,
                    "name": self.graph.nodes[phenotype_id]["name"]
                })
        return phenotypes
    
    def find_similar_diseases(
        self,
        phenotype_ids: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find diseases with similar phenotypes."""
        scores = {}
        
        for disease_id in self.graph.nodes():
            if self.graph.nodes[disease_id]["type"] != "disease":
                continue
            
            # Calculate Jaccard similarity
            disease_phenotypes = set(
                target for _, target in self.graph.edges(disease_id)
                if self.graph.nodes[target]["type"] == "phenotype"
            )
            query_phenotypes = set(phenotype_ids)
            
            if not disease_phenotypes or not query_phenotypes:
                continue
            
            similarity = len(disease_phenotypes & query_phenotypes) / len(disease_phenotypes | query_phenotypes)
            scores[disease_id] = similarity
        
        # Sort by similarity score
        sorted_diseases = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            {
                "orpha_code": disease_id,
                "name": self.graph.nodes[disease_id]["name"],
                "similarity_score": score,
                "phenotypes": self.get_disease_phenotypes(disease_id)
            }
            for disease_id, score in sorted_diseases
        ]
    
    def get_disease_path(
        self,
        source_disease: str,
        target_disease: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Find the shortest path between two diseases through phenotypes."""
        try:
            path = nx.shortest_path(
                self.graph,
                source=source_disease,
                target=target_disease
            )
            
            return [
                {
                    "id": node_id,
                    "type": self.graph.nodes[node_id]["type"],
                    "name": self.graph.nodes[node_id]["name"]
                }
                for node_id in path
            ]
        except nx.NetworkXNoPath:
            return None 

    def add_article(self, article: PubMedArticle) -> None:
        """
        Add an article node and its relationships to the graph.
        
        Args:
            article: PubMedArticle instance to add
        """
        node = article.to_dict()
        node['type'] = 'article'
        # Ensure 'name' exists (use title)
        if 'name' not in node or not node['name']:
            node['name'] = node.get('title', node.get('pmid', 'Unknown Article'))
        self.graph.add_node(node['pmid'], **node)
        self.article_nodes[node['pmid']] = node

        # Add MeSH term relationships
        for mesh_term in article.mesh_terms:
            mesh_id = f"mesh_{mesh_term.get('id', '')}"
            if mesh_id not in self.mesh_term_nodes:
                self.mesh_term_nodes[mesh_id] = {
                    "type": "mesh_term",
                    "name": mesh_term.get("name", ""),
                    "qualifier": mesh_term.get("qualifier", "")
                }
                self.graph.add_node(mesh_id, **self.mesh_term_nodes[mesh_id])
            self.graph.add_edge(node['pmid'], mesh_id, relationship="has_mesh_term")

        # Add chemical relationships
        for chemical in article.chemicals:
            chem_id = f"chemical_{chemical.get('id', '')}"
            if chem_id not in self.chemical_nodes:
                self.chemical_nodes[chem_id] = {
                    "type": "chemical",
                    "name": chemical.get("name", ""),
                    "registry_number": chemical.get("registry_number", "")
                }
                self.graph.add_node(chem_id, **self.chemical_nodes[chem_id])
            self.graph.add_edge(node['pmid'], chem_id, relationship="mentions_chemical")

    def link_disease_to_article(self, disease_id: str, article_id: str, relationship: str = "discussed_in") -> None:
        """
        Create a relationship between a disease and an article.
        
        Args:
            disease_id: ID of the disease node
            article_id: ID of the article node
            relationship: Type of relationship between the nodes
        """
        disease_node = f"disease_{disease_id}"
        article_node = f"article_{article_id}"
        
        if disease_node in self.disease_nodes and article_node in self.article_nodes:
            self.graph.add_edge(disease_node, article_node, relationship=relationship)
        else:
            logger.warning(f"Could not create relationship: one or both nodes not found")

    def get_disease_articles(self, disease_id: str) -> List[Dict[str, Any]]:
        """
        Get all articles related to a disease.
        
        Args:
            disease_id: ID of the disease
            
        Returns:
            List of article data dictionaries
        """
        disease_node = f"disease_{disease_id}"
        if disease_node not in self.disease_nodes:
            return []
            
        articles = []
        for neighbor in self.graph.neighbors(disease_node):
            if self.graph.nodes[neighbor]["type"] == "article":
                articles.append(self.graph.nodes[neighbor])
        return articles

    def get_article_diseases(self, article_id: str) -> List[Dict[str, Any]]:
        """
        Get all diseases mentioned in an article.
        
        Args:
            article_id: ID of the article
            
        Returns:
            List of disease data dictionaries
        """
        article_node = f"article_{article_id}"
        if article_node not in self.article_nodes:
            return []
            
        diseases = []
        for neighbor in self.graph.neighbors(article_node):
            if self.graph.nodes[neighbor]["type"] == "disease":
                diseases.append(self.graph.nodes[neighbor])
        return diseases

    def search_diseases(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for diseases by name or other attributes.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching disease data dictionaries
        """
        query = query.lower()
        matches = []
        
        for node_id, node_data in self.disease_nodes.items():
            if (query in node_data["name"].lower() or
                any(query in str(value).lower() for value in node_data.values())):
                matches.append(node_data)
                
        return matches

    def search_articles(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for articles by title, abstract, or other attributes.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching article data dictionaries
        """
        query = query.lower()
        matches = []
        
        for node_id, node_data in self.article_nodes.items():
            if (query in node_data["title"].lower() or
                (node_data["abstract"] and query in node_data["abstract"].lower()) or
                any(query in str(value).lower() for value in node_data.values())):
                matches.append(node_data)
                
        return matches

    def get_disease_network(self, disease_id: str, depth: int = 2) -> nx.DiGraph:
        """
        Get the subgraph of related nodes around a disease.
        
        Args:
            disease_id: ID of the disease
            depth: Maximum distance from the disease node
            
        Returns:
            NetworkX DiGraph containing the disease network
        """
        disease_node = f"disease_{disease_id}"
        if disease_node not in self.disease_nodes:
            return nx.DiGraph()
            
        # Get all nodes within the specified depth
        nodes = {disease_node}
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.graph.neighbors(node))
            nodes.update(new_nodes)
            
        return self.graph.subgraph(nodes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the knowledge graph to a dictionary representation."""
        return {
            "nodes": {
                "diseases": self.disease_nodes,
                "articles": self.article_nodes,
                "mesh_terms": self.mesh_term_nodes,
                "chemicals": self.chemical_nodes
            },
            "edges": [(u, v, d) for u, v, d in self.graph.edges(data=True)]
        }

    def save_graph(self, filepath: str) -> None:
        """
        Save the graph to a file.
        
        Args:
            filepath: Path to save the graph
        """
        # Convert graph to dictionary format
        graph_data = {
            "nodes": [
                {
                    "id": node_id,
                    **self.graph.nodes[node_id]
                }
                for node_id in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **d
                }
                for u, v, d in self.graph.edges(data=True)
            ]
        }
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
        logger.info(f"Saved knowledge graph to {filepath}")

    @classmethod
    def load_graph(cls, filepath: str) -> 'KnowledgeGraph':
        """
        Load a graph from a file.
        
        Args:
            filepath: Path to the saved graph file
            
        Returns:
            KnowledgeGraph instance with loaded data
        """
        kg = cls()
        
        # Load from JSON file
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        
        # Reconstruct graph
        for node in graph_data["nodes"]:
            # Ensure 'name' exists
            if 'name' not in node or not node['name']:
                if node.get('type') == 'article':
                    node['name'] = node.get('title', node.get('pmid', 'Unknown Article'))
                elif node.get('type') == 'disease':
                    node['name'] = node.get('disease_id', 'Unknown Disease')
                else:
                    node['name'] = node.get('id', 'Unknown')
            kg.graph.add_node(node.get('disease_id') or node.get('pmid') or node.get('mesh_id') or node.get('chem_id') or node.get('id'), **node)
            
            # Update node dictionaries
            if node["type"] == "disease":
                kg.disease_nodes[node['disease_id']] = node
            elif node["type"] == "article":
                kg.article_nodes[node['pmid']] = node
            elif node["type"] == "mesh_term":
                kg.mesh_term_nodes[node['mesh_id']] = node
            elif node["type"] == "chemical":
                kg.chemical_nodes[node['chem_id']] = node
        
        for edge in graph_data["edges"]:
            source = edge.pop("source")
            target = edge.pop("target")
            kg.graph.add_edge(source, target, **edge)
        
        # Recompute embeddings
        kg._compute_node_embeddings()
        
        logger.info(f"Loaded knowledge graph from {filepath}")
        return kg 