#!/usr/bin/env python3
"""
RAG-Ready Knowledge Graph Builder
Creates structured data suitable for LLM retrieval and generation.
"""

import json
import os
import numpy as np
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
import uuid
import logging

from app.data.pubmed_collector import PubMedArticle
from app.data.orphanet_collector import OrphanetDisease
from app.data.fda_collector import FDACollector

class RAGReadyGraphBuilder:
    """Builds RAG-ready knowledge graphs with structured data for LLM consumption."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the RAG-ready graph builder."""
        print(f"Initializing RAG-Ready Graph Builder with {model_name}")
        self.model = SentenceTransformer(model_name)
        self.faiss_index = None
        self.text_chunks = []
        self.chunk_embeddings = {}
        self.entity_nodes = {}
        self.relationship_edges = {}
        
        # Initialize collectors
        self.pubmed_collector = None
        self.orphanet_collector = None
        self.fda_collector = FDACollector()
        
    def setup_collectors(self):
        """Setup collectors if API keys are available."""
        if os.getenv('PUBMED_API_KEY'):
            self.pubmed_collector = PubMedArticle
            print("✓ PubMed collector enabled")
        
        if os.getenv('ORPHANET_API_KEY'):
            self.orphanet_collector = OrphanetDisease
            print("✓ Orphanet collector enabled")
        
        print("✓ FDA collector enabled")
    
    def create_text_chunks(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Create overlapping text chunks for RAG."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                last_semicolon = chunk.rfind(';')
                last_comma = chunk.rfind(',')
                
                break_point = max(last_period, last_semicolon, last_comma)
                if break_point > start + chunk_size // 2:  # Only break if it's not too early
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def extract_pubmed_chunks(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Extract PubMed articles and create RAG-ready chunks."""
        if not self.pubmed_collector:
            print("PubMed collector not available")
            return []
        
        print(f"Extracting PubMed chunks for: {query}")
        chunks = []
        
        try:
            articles = self.pubmed_collector.search(query, max_results=max_results, api_key=os.getenv('PUBMED_API_KEY'))
            
            for article in articles:
                # Create rich text content
                content_parts = []
                content_parts.append(f"Title: {article.title}")
                
                if article.authors:
                    content_parts.append(f"Authors: {', '.join(article.authors[:5])}")
                
                if article.journal:
                    content_parts.append(f"Journal: {article.journal}")
                
                if article.publication_date:
                    content_parts.append(f"Publication Date: {article.publication_date}")
                
                if article.abstract:
                    content_parts.append(f"Abstract: {article.abstract}")
                
                # Add MeSH terms if available
                if article.mesh_terms:
                    content_parts.append(f"MeSH Terms: {', '.join(article.mesh_terms[:10])}")
                
                # Add chemicals if available
                if article.chemicals:
                    content_parts.append(f"Chemicals: {', '.join(article.chemicals[:10])}")
                
                full_content = "\n".join(content_parts)
                
                # Create chunks
                text_chunks = self.create_text_chunks(full_content, chunk_size=800, overlap=150)
                
                for i, chunk in enumerate(text_chunks):
                    chunk_id = f"pubmed_{article.pmid}_chunk_{i}"
                    
                    chunk_data = {
                        "id": chunk_id,
                        "content": chunk,
                        "type": "scientific_article",
                        "source": "pubmed",
                        "metadata": {
                            "pmid": article.pmid,
                            "title": article.title,
                            "authors": article.authors,
                            "journal": article.journal,
                            "publication_date": article.publication_date,
                            "chunk_index": i,
                            "total_chunks": len(text_chunks),
                            "mesh_terms": article.mesh_terms,
                            "chemicals": article.chemicals,
                            "keywords": self._extract_keywords(chunk),
                            "entities": self._extract_entities(chunk)
                        },
                        "rag_metadata": {
                            "retrieval_score": 0.0,
                            "relevance_tags": [],
                            "question_types": ["what", "how", "why", "when", "where"],
                            "content_type": "research_article"
                        }
                    }
                    
                    chunks.append(chunk_data)
                    
                    # Store as entity node
                    self.entity_nodes[chunk_id] = {
                        "id": chunk_id,
                        "type": "scientific_article",
                        "source": "pubmed",
                        "name": f"{article.title[:50]}... (Chunk {i+1})",
                        "content": chunk,
                        "metadata": chunk_data["metadata"]
                    }
        
        except Exception as e:
            print(f"PubMed extraction failed: {e}")
        
        print(f"Created {len(chunks)} PubMed chunks")
        return chunks
    
    def extract_orphanet_chunks(self, query: str) -> List[Dict[str, Any]]:
        """Extract Orphanet disease data and create RAG-ready chunks."""
        if not self.orphanet_collector:
            print("Orphanet collector not available")
            return []
        
        print(f"Extracting Orphanet chunks for: {query}")
        chunks = []
        
        try:
            diseases = self.orphanet_collector.search_by_name(query, api_key=os.getenv('ORPHANET_API_KEY'))
            
            for disease in diseases:
                # Create comprehensive disease content
                content_parts = []
                content_parts.append(f"Disease: {disease.name}")
                
                # Add definition
                raw_data = disease.metadata.get("raw_data", {})
                data = raw_data.get("data", {})
                results = data.get("results", {})
                summary_info = results.get("SummaryInformation", [])
                
                for summary in summary_info:
                    definition = summary.get("Definition", "")
                    if definition:
                        content_parts.append(f"Definition: {definition}")
                
                # Add prevalence
                if disease.prevalence:
                    content_parts.append(f"Prevalence: {disease.prevalence}")
                
                # Add inheritance
                if disease.inheritance:
                    content_parts.append(f"Inheritance: {', '.join(disease.inheritance)}")
                
                # Add age of onset
                if disease.age_of_onset:
                    content_parts.append(f"Age of Onset: {', '.join(disease.age_of_onset)}")
                
                # Add phenotypes
                if disease.phenotypes:
                    pheno_names = [p.get("name", "") for p in disease.phenotypes[:10]]
                    if any(pheno_names):
                        content_parts.append(f"Phenotypes: {', '.join(pheno_names)}")
                
                # Add medical specialties
                if disease.medical_specialties:
                    content_parts.append(f"Medical Specialties: {', '.join(disease.medical_specialties)}")
                
                # Add ICD codes
                if disease.icd10_codes:
                    content_parts.append(f"ICD-10 Codes: {', '.join(disease.icd10_codes)}")
                
                # Add OMIM IDs
                if disease.omim_ids:
                    content_parts.append(f"OMIM IDs: {', '.join(disease.omim_ids)}")
                
                full_content = "\n".join(content_parts)
                
                # Create chunks
                text_chunks = self.create_text_chunks(full_content, chunk_size=600, overlap=100)
                
                for i, chunk in enumerate(text_chunks):
                    chunk_id = f"orphanet_{disease.disease_id}_chunk_{i}"
                    
                    chunk_data = {
                        "id": chunk_id,
                        "content": chunk,
                        "type": "disease_information",
                        "source": "orphanet",
                        "metadata": {
                            "disease_id": disease.disease_id,
                            "disease_name": disease.name,
                            "prevalence": disease.prevalence,
                            "inheritance": disease.inheritance,
                            "age_of_onset": disease.age_of_onset,
                            "phenotypes": disease.phenotypes,
                            "medical_specialties": disease.medical_specialties,
                            "icd10_codes": disease.icd10_codes,
                            "omim_ids": disease.omim_ids,
                            "chunk_index": i,
                            "total_chunks": len(text_chunks),
                            "keywords": self._extract_keywords(chunk),
                            "entities": self._extract_entities(chunk)
                        },
                        "rag_metadata": {
                            "retrieval_score": 0.0,
                            "relevance_tags": ["clinical", "disease", "medical"],
                            "question_types": ["what", "symptoms", "treatment", "diagnosis"],
                            "content_type": "clinical_information"
                        }
                    }
                    
                    chunks.append(chunk_data)
                    
                    # Store as entity node
                    self.entity_nodes[chunk_id] = {
                        "id": chunk_id,
                        "type": "disease_information",
                        "source": "orphanet",
                        "name": f"{disease.name} (Chunk {i+1})",
                        "content": chunk,
                        "metadata": chunk_data["metadata"]
                    }
        
        except Exception as e:
            print(f"Orphanet extraction failed: {e}")
        
        print(f"Created {len(chunks)} Orphanet chunks")
        return chunks
    
    def extract_fda_chunks(self, search_terms: List[str], max_results: int = 3) -> List[Dict[str, Any]]:
        """Extract FDA drug data and create RAG-ready chunks."""
        print(f"Extracting FDA chunks with terms: {search_terms}")
        chunks = []
        
        for term in search_terms:
            try:
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
                                
                                # Create comprehensive drug content
                                content_parts = []
                                content_parts.append(f"Drug: {drug_name}")
                                
                                # Add active ingredients
                                active_ingredients = row['openfda'].get('substance_name', [])
                                if active_ingredients:
                                    content_parts.append(f"Active Ingredients: {', '.join(active_ingredients)}")
                                
                                # Add description
                                description = row.get('description', [])
                                if description:
                                    content_parts.append(f"Description: {description[0]}")
                                
                                # Add indications
                                indications = row.get('indications_and_usage', [])
                                if indications:
                                    content_parts.append(f"Indications and Usage: {indications[0]}")
                                
                                # Add dosage
                                dosage = row.get('dosage_and_administration', [])
                                if dosage:
                                    content_parts.append(f"Dosage and Administration: {dosage[0]}")
                                
                                # Add warnings
                                warnings = row.get('warnings', [])
                                if warnings:
                                    content_parts.append(f"Warnings: {warnings[0]}")
                                
                                # Add adverse reactions
                                adverse_reactions = row.get('adverse_reactions', [])
                                if adverse_reactions:
                                    content_parts.append(f"Adverse Reactions: {adverse_reactions[0]}")
                                
                                # Add drug interactions
                                interactions = row.get('drug_interactions', [])
                                if interactions:
                                    content_parts.append(f"Drug Interactions: {interactions[0]}")
                                
                                full_content = "\n".join(content_parts)
                                
                                # Create chunks
                                text_chunks = self.create_text_chunks(full_content, chunk_size=700, overlap=150)
                                
                                for i, chunk in enumerate(text_chunks):
                                    chunk_id = f"fda_{drug_name.lower().replace(' ', '_').replace('-', '_')}_chunk_{i}"
                                    
                                    chunk_data = {
                                        "id": chunk_id,
                                        "content": chunk,
                                        "type": "drug_information",
                                        "source": "fda",
                                        "metadata": {
                                            "drug_name": drug_name,
                                            "active_ingredients": active_ingredients,
                                            "description": description[0] if description else "",
                                            "indications": indications[0] if indications else "",
                                            "dosage": dosage[0] if dosage else "",
                                            "warnings": warnings[0] if warnings else "",
                                            "adverse_reactions": adverse_reactions[:3] if adverse_reactions else [],
                                            "drug_interactions": interactions[:3] if interactions else [],
                                            "search_term": term,
                                            "chunk_index": i,
                                            "total_chunks": len(text_chunks),
                                            "keywords": self._extract_keywords(chunk),
                                            "entities": self._extract_entities(chunk)
                                        },
                                        "rag_metadata": {
                                            "retrieval_score": 0.0,
                                            "relevance_tags": ["drug", "medication", "treatment", "side_effects"],
                                            "question_types": ["dosage", "side_effects", "interactions", "indications"],
                                            "content_type": "drug_label"
                                        }
                                    }
                                    
                                    chunks.append(chunk_data)
                                    
                                    # Store as entity node
                                    self.entity_nodes[chunk_id] = {
                                        "id": chunk_id,
                                        "type": "drug_information",
                                        "source": "fda",
                                        "name": f"{drug_name} (Chunk {i+1})",
                                        "content": chunk,
                                        "metadata": chunk_data["metadata"]
                                    }
                            
                            break  # Stop if we found results
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"FDA extraction failed for term '{term}': {e}")
        
        print(f"Created {len(chunks)} FDA chunks")
        return chunks
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for RAG retrieval."""
        # Simple keyword extraction (in practice, you'd use NLP)
        keywords = []
        text_lower = text.lower()
        
        # Medical keywords
        medical_terms = [
            'treatment', 'therapy', 'medication', 'drug', 'antibiotic', 'vaccine',
            'prevention', 'management', 'intervention', 'pharmacological',
            'symptoms', 'diagnosis', 'prognosis', 'etiology', 'pathophysiology',
            'clinical', 'therapeutic', 'dosage', 'administration', 'contraindications'
        ]
        
        for term in medical_terms:
            if term in text_lower:
                keywords.append(term)
        
        return keywords
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text for RAG retrieval."""
        # Simple entity extraction (in practice, you'd use NER)
        entities = []
        
        # Look for capitalized phrases (potential entities)
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Check if it's part of a multi-word entity
                entity = word
                j = i + 1
                while j < len(words) and words[j][0].isupper():
                    entity += " " + words[j]
                    j += 1
                if entity not in entities:
                    entities.append(entity)
        
        return entities[:10]  # Limit to top 10
    
    def build_faiss_index(self, chunks: List[Dict[str, Any]]):
        """Build FAISS index for RAG retrieval, skipping empty/duplicate chunks."""
        print(f"Building FAISS index for {len(chunks)} chunks")
        if not chunks:
            print("No chunks to index")
            return
        # Filter out chunks with empty content or duplicate IDs
        valid_chunks = []
        ids_seen = set()
        skipped_empty = 0
        skipped_duplicate = 0
        for i, chunk in enumerate(chunks):
            cid = chunk["id"]
            content = chunk["content"].strip() if chunk["content"] else ""
            if not cid:
                print(f"Chunk {i} has empty ID! Skipping.")
                skipped_empty += 1
                continue
            if cid in ids_seen:
                print(f"Duplicate chunk ID: {cid} (skipping)")
                skipped_duplicate += 1
                continue
            if not content:
                print(f"Chunk {cid} has empty content! Skipping.")
                skipped_empty += 1
                continue
            ids_seen.add(cid)
            valid_chunks.append(chunk)
        print(f"Valid chunks for embedding: {len(valid_chunks)} (skipped {skipped_empty} empty, {skipped_duplicate} duplicates)")
        if not valid_chunks:
            print("No valid chunks to index!")
            return
        # Compute embeddings for all valid chunks
        texts = [chunk["content"] for chunk in valid_chunks]
        embeddings = self.model.encode(texts)
        # Store chunk information
        self.chunk_embeddings = {}  # Reset to avoid stale data
        for i, chunk in enumerate(valid_chunks):
            chunk_id = chunk["id"]
            self.chunk_embeddings[chunk_id] = embeddings[i]
        print(f"Chunk embeddings keys: {len(self.chunk_embeddings)}")
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        print(f"FAISS index built with {len(valid_chunks)} chunks, dimension {dimension}")
    
    def keyword_fallback(self, query, chunks, top_k=5):
        query_words = set(query.lower().split())
        scored = []
        for chunk in chunks:
            if not chunk or not chunk.get('content'):
                continue
            content = chunk['content'].lower()
            score = sum(1 for word in query_words if word in content)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [chunk for score, chunk in scored[:top_k]]

    def retrieve_relevant_chunks(self, query: str, top_k: int = 10000) -> List[Dict[str, Any]]:
        """Return all chunks for brute-force RAG (feed all to LLM)."""
        chunk_ids = list(self.chunk_embeddings.keys())
        all_chunks = [self._get_chunk_data(cid) for cid in chunk_ids if self._get_chunk_data(cid)]
        print(f"[DEBUG] Feeding {len(all_chunks)} chunks to the LLM.")
        return all_chunks
    
    def _get_chunk_data(self, chunk_id: str) -> Dict[str, Any]:
        """Get chunk data by ID."""
        # This would need to be implemented based on how you store chunk data
        # For now, return basic info
        if chunk_id in self.entity_nodes:
            node = self.entity_nodes[chunk_id]
            return {
                "id": chunk_id,
                "content": node["content"],
                "type": node["type"],
                "source": node["source"],
                "metadata": node["metadata"]
            }
        return None
    
    def build_rag_ready_graph(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Build a RAG-ready knowledge graph."""
        print(f"\n{'='*60}")
        print(f"Building RAG-Ready Knowledge Graph for: {query}")
        print(f"{'='*60}")
        
        # Setup collectors
        self.setup_collectors()
        
        all_chunks = []
        
        # Step 1: Extract PubMed chunks
        print(f"\n1. Extracting PubMed chunks...")
        pubmed_chunks = self.extract_pubmed_chunks(query, max_results)
        all_chunks.extend(pubmed_chunks)
        
        # Step 2: Extract Orphanet chunks
        print(f"\n2. Extracting Orphanet chunks...")
        orphanet_chunks = self.extract_orphanet_chunks(query)
        all_chunks.extend(orphanet_chunks)
        
        # Step 3: Extract search terms for FDA
        print(f"\n3. Extracting search terms for FDA...")
        search_terms = set()
        for chunk in all_chunks:
            keywords = chunk["metadata"].get("keywords", [])
            search_terms.update(keywords)
        search_terms.add(query.lower())
        
        # Step 4: Extract FDA chunks
        print(f"\n4. Extracting FDA chunks...")
        fda_chunks = self.extract_fda_chunks(list(search_terms), max_results)
        all_chunks.extend(fda_chunks)
        
        # Step 5: Build FAISS index
        print(f"\n5. Building FAISS index...")
        self.build_faiss_index(all_chunks)
        
        # Step 6: Create RAG-ready structure
        print(f"\n6. Creating RAG-ready structure...")
        rag_structure = self._create_rag_structure(all_chunks)
        
        # Step 7: Save results
        print(f"\n7. Saving results...")
        self._save_rag_results(rag_structure, all_chunks)
        
        return rag_structure
    
    def _create_rag_structure(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create RAG-ready structure."""
        
        # Group chunks by source
        chunks_by_source = {}
        for chunk in chunks:
            source = chunk["source"]
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)
        
        # Create RAG-ready structure
        rag_structure = {
            "metadata": {
                "total_chunks": len(chunks),
                "chunks_by_source": {source: len(chunk_list) for source, chunk_list in chunks_by_source.items()},
                "created_at": datetime.now().isoformat(),
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size": "variable",
                "overlap": "variable"
            },
            "chunks": chunks,
            "entity_nodes": self.entity_nodes,
            "faiss_index_info": {
                "total_vectors": len(chunks),
                "dimension": 384,
                "index_type": "IndexFlatIP"
            },
            "retrieval_config": {
                "default_top_k": 5,
                "similarity_threshold": 0.6,
                "reranking_enabled": False
            }
        }
        
        return rag_structure
    
    def _save_rag_results(self, rag_structure: Dict[str, Any], chunks: List[Dict[str, Any]]):
        """Save RAG-ready results."""
        
        # Save RAG structure
        with open("rag_ready_graph.json", "w") as f:
            json.dump(rag_structure, f, indent=2)
        
        # Save chunks for easy access
        with open("rag_chunks.json", "w") as f:
            json.dump(chunks, f, indent=2)
        
        # Save entity nodes
        with open("rag_entity_nodes.json", "w") as f:
            json.dump(self.entity_nodes, f, indent=2)
        
        # Create RAG retrieval examples
        example_queries = [
            "What are the symptoms of cholera?",
            "How is cholera treated?",
            "What drugs are used for cholera treatment?",
            "What is the prevalence of cholera?",
            "What are the adverse effects of cholera medications?"
        ]
        
        retrieval_examples = []
        for query in example_queries:
            relevant_chunks = self.retrieve_relevant_chunks(query, top_k=3)
            retrieval_examples.append({
                "query": query,
                "relevant_chunks": [
                    {
                        "id": chunk["id"],
                        "content": chunk["content"][:200] + "...",
                        "type": chunk["type"],
                        "source": chunk["source"],
                        "retrieval_score": chunk.get("retrieval_score", 0.0)
                    }
                    for chunk in relevant_chunks
                ]
            })
        
        with open("rag_retrieval_examples.json", "w") as f:
            json.dump(retrieval_examples, f, indent=2)
        
        print("RAG-ready results saved to:")
        print("  - rag_ready_graph.json (complete RAG structure)")
        print("  - rag_chunks.json (all text chunks)")
        print("  - rag_entity_nodes.json (entity information)")
        print("  - rag_retrieval_examples.json (retrieval examples)")

def main():
    """Main function to demonstrate RAG-ready graph building."""
    print("RAG-Ready Knowledge Graph Builder")
    print("Creating structured data suitable for LLM retrieval and generation")
    print("=" * 80)
    
    # Initialize builder
    builder = RAGReadyGraphBuilder()
    
    # Build RAG-ready graph
    query = "Cholera"
    rag_structure = builder.build_rag_ready_graph(query, max_results=3)
    
    # Display results
    print(f"\n{'='*60}")
    print("RAG-READY GRAPH COMPLETE!")
    print(f"{'='*60}")
    
    metadata = rag_structure["metadata"]
    print(f"\n📊 RAG Structure Summary:")
    print(f"   Total Chunks: {metadata['total_chunks']}")
    print(f"   Chunks by Source:")
    for source, count in metadata['chunks_by_source'].items():
        print(f"     {source.capitalize()}: {count}")
    print(f"   Embedding Model: {metadata['embedding_model']}")
    
    # Test retrieval
    print(f"\n🔍 Testing RAG Retrieval:")
    test_query = "What are the symptoms of cholera?"
    relevant_chunks = builder.retrieve_relevant_chunks(test_query, top_k=3)
    
    print(f"Query: {test_query}")
    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"  {i}. {chunk['type']} ({chunk['source']}) - Score: {chunk.get('retrieval_score', 0.0):.3f}")
        print(f"     {chunk['content'][:100]}...")
    
    print(f"\n📁 Files saved:")
    print(f"   - rag_ready_graph.json")
    print(f"   - rag_chunks.json")
    print(f"   - rag_entity_nodes.json")
    print(f"   - rag_retrieval_examples.json")

if __name__ == "__main__":
    main() 