import logging
import os
import json
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.core.knowledge_graph import KnowledgeGraph
from app.core.entity_extractor import MedicalEntityExtractor
from app.core.entity_extractor_llm import LLMEntityExtractor
from app.rag.rag_engine import RAGEngine
from app.data.pubmed_collector import PubMedArticle
from app.data.orphanet_collector import OrphanetDisease
from app.data.fda_collector import FDACollector
from rag_ready_graph_builder import RAGReadyGraphBuilder
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical RAG Assistant",
    description="A comprehensive medical assistant using RAG with knowledge graphs",
    version="1.0.0"
)

# Serve static files (UI) at /static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Serve index.html at root
@app.get("/")
def serve_index():
    return FileResponse(os.path.join("static", "index.html"))

# Global instances
knowledge_graph = KnowledgeGraph()
entity_extractor = MedicalEntityExtractor()
llm_entity_extractor = LLMEntityExtractor()  # DeepSeek LLM extractor
rag_engine = RAGEngine(knowledge_graph)
pubmed_collector = None
orphanet_collector = None
fda_collector = FDACollector()

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    max_context: int = 5
    build_graph: bool = True  # Whether to build knowledge graph from query
    use_llm: bool = False  # Whether to use LLM for entity extraction
    use_rag_ready: bool = False  # Whether to use RAG-ready graph with LLM

class DiseaseSearchRequest(BaseModel):
    disease_name: str
    max_results: int = 5

class DrugSearchRequest(BaseModel):
    drug_name: str
    max_results: int = 5

class EntityExtractionRequest(BaseModel):
    text: str
    enrich_with_apis: bool = True
    use_llm: bool = False  # Whether to use LLM instead of regex/spaCy

class Response(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    confidence: float = 0.0
    follow_up_question: str | None = None
    entities_extracted: List[Dict[str, Any]] = []
    relationships_built: List[Dict[str, Any]] = []
    method: str = ""

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global pubmed_collector, orphanet_collector
    
    logger.info("Initializing Medical RAG Assistant...")
    
    # Initialize collectors
    try:
        # PubMed collector - doesn't require API key for basic searches
        pubmed_collector = PubMedArticle
        logger.info("PubMed collector initialized")
            
        if settings.orphanet_api_key:
            orphanet_collector = OrphanetDisease
            entity_extractor.set_orphanet_collector(OrphanetDisease)
            logger.info("Orphanet collector initialized")
        else:
            logger.warning("Orphanet API key not found - Orphanet features disabled")
            
        logger.info("FDA collector initialized")
        logger.info("DeepSeek LLM entity extractor initialized")
        
    except Exception as e:
        logger.error(f"Error initializing collectors: {e}")
    
    # Load existing knowledge graph if available
    try:
        if os.path.exists("knowledge_graph.json"):
            knowledge_graph = KnowledgeGraph.load_graph("knowledge_graph.json")
            logger.info("Loaded existing knowledge graph")
        else:
            logger.info("No existing knowledge graph found - starting fresh")
    except Exception as e:
        logger.error(f"Error loading knowledge graph: {e}")

@app.post("/query", response_model=Response)
async def answer_question(request: QueryRequest):
    """Answer a medical question using RAG and optionally build knowledge graph."""
    try:
        entities_extracted = []
        relationships_built = []
        
        # Build knowledge graph from question if requested
        if request.build_graph:
            try:
                if request.use_llm:
                    # Use DeepSeek LLM for entity extraction
                    result = llm_entity_extractor.extract_entities_with_relationships(request.query)
                    entities_extracted = result['entities']
                    relationships_built = result['relationships']
                    
                    # Add entities and relationships to knowledge graph
                    knowledge_graph.add_dynamic_entities(entities_extracted)
                    knowledge_graph.add_dynamic_relationships(relationships_built)
                    
                    logger.info(f"LLM extracted {len(entities_extracted)} entities and {len(relationships_built)} relationships")
                else:
                    # Use traditional extractor
                    result = knowledge_graph.process_medical_question(request.query, entity_extractor)
                    entities_extracted = result['entities']
                    relationships_built = result['relationships']
                    logger.info(f"Traditional extractor found {result['total_entities']} entities and {result['total_relationships']} relationships")
            except Exception as e:
                logger.warning(f"Failed to build knowledge graph from question: {e}")
        
        # Get RAG answer (now includes confidence and follow-up question)
        result = rag_engine.answer_question(request.query, use_rag_ready=request.use_rag_ready)
        
        return Response(
            answer=result['answer'],
            context=result['context'],
            confidence=result.get('confidence', 0.0),
            follow_up_question=result.get('follow_up_question'),
            entities_extracted=entities_extracted,
            relationships_built=relationships_built,
            method="rag" if request.use_rag_ready else "traditional"
        )
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-rag-ready", response_model=Response)
async def answer_question_rag_ready(request: QueryRequest):
    """Answer a medical question using RAG-ready graph with LLM."""
    try:
        # Check if RAG-ready graph exists
        if not os.path.exists("rag_ready_graph.json"):
            raise HTTPException(
                status_code=404, 
                detail="RAG-ready graph not found. Please build it first using /rag-ready/build-graph"
            )
        
        # Get RAG-ready answer with LLM
        result = rag_engine.answer_question(request.query, use_rag_ready=True)
        
        return Response(
            answer=result['answer'],
            context=result['context'],
            confidence=result.get('confidence', 0.0),
            follow_up_question=result.get('follow_up_question'),
            entities_extracted=[],
            relationships_built=[],
            method="rag_ready_llm"
        )
    except Exception as e:
        logger.error(f"Error answering question with RAG-ready: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-entities")
async def extract_entities(request: EntityExtractionRequest):
    """Extract medical entities from text using either traditional or LLM methods."""
    try:
        if request.use_llm:
            # Use DeepSeek LLM for extraction
            result = llm_entity_extractor.extract_entities_with_relationships(request.text)
            entities = result['entities']
            relationships = result['relationships']
            
            # Enrich with API data if requested
            if request.enrich_with_apis:
                # Note: LLM extraction already provides high-quality entities
                # API enrichment could be added here if needed
                pass
        else:
            # Use traditional extractor
            extracted_entities = entity_extractor.extract_entities(request.text)
            
            # Enrich with API data if requested
            if request.enrich_with_apis:
                enriched_entities = entity_extractor.enrich_entities(extracted_entities)
            else:
                enriched_entities = [
                    {
                        'text': entity.text,
                        'type': entity.type,
                        'confidence': entity.confidence,
                        'metadata': entity.metadata or {}
                    }
                    for entity in extracted_entities
                ]
            
            entities = enriched_entities
            relationships = entity_extractor.build_relationships(extracted_entities, request.text)
        
        return {
            "entities": entities,
            "relationships": relationships,
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "method": "llm" if request.use_llm else "traditional"
        }
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-entities-llm")
async def extract_entities_llm(request: EntityExtractionRequest):
    """Extract medical entities using DeepSeek LLM specifically."""
    try:
        result = llm_entity_extractor.extract_entities_with_relationships(request.text)
        
        return {
            "entities": result['entities'],
            "relationships": result['relationships'],
            "total_entities": len(result['entities']),
            "total_relationships": len(result['relationships']),
            "method": "llm"
        }
    except Exception as e:
        logger.error(f"Error extracting entities with LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/build-graph-from-text")
async def build_graph_from_text(request: EntityExtractionRequest):
    """Build knowledge graph from text by extracting entities and relationships using LLM and collector enrichment only."""
    try:
        # Use LLM for extraction
        result = llm_entity_extractor.extract_entities_with_relationships(request.text)
        entities = result['entities']
        relationships = result['relationships']
        # Add to knowledge graph
        knowledge_graph.add_dynamic_entities(entities)
        knowledge_graph.add_dynamic_relationships(relationships)
        # Save the updated graph
        knowledge_graph.save_graph("knowledge_graph.json")
        return {
            "message": "Knowledge graph updated successfully",
            "entities_added": len(entities),
            "relationships_added": len(relationships),
            "entities": entities,
            "relationships": relationships,
            "method": "llm"
        }
    except Exception as e:
        logger.error(f"Error building graph from text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/entity-network/{entity_type}/{entity_name}")
async def get_entity_network(entity_type: str, entity_name: str, depth: int = 2):
    """Get the network around a specific entity."""
    try:
        network = knowledge_graph.get_entity_network(entity_name, entity_type, depth)
        return network
    except Exception as e:
        logger.error(f"Error getting entity network: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/diseases")
async def search_diseases(request: DiseaseSearchRequest):
    """Search for diseases using all collectors, always getting OrphaCode and full Orphanet info."""
    try:
        # --- Orphanet: Always get OrphaCode and full info ---
        orphanet_results = []
        diseases = OrphanetDisease.search_by_name(request.disease_name)
        for disease in diseases:
            orphanet_results.append({
                "orpha_code": disease.disease_id,
                "name": disease.name,
                "definition": (
                    disease.metadata.get("raw_data", {})
                    .get("data", {})
                    .get("results", {})
                    .get("SummaryInformation", [{}])[0].get("Definition")
                ),
                "orphanet_url": (
                    disease.metadata.get("raw_data", {})
                    .get("data", {})
                    .get("results", {})
                    .get("OrphanetURL")
                ),
            })
        # --- PubMed ---
        graph_results = knowledge_graph.search_diseases(request.disease_name)
        api_results = []
        if pubmed_collector:
            articles = pubmed_collector.search(request.disease_name, max_results=request.max_results)
            for article in articles:
                api_results.append({
                    "pmid": article.pmid,
                    "title": article.title,
                    "journal": article.journal,
                    "source": "PubMed"
                })
        return {
            "orphanet_results": orphanet_results,
            "graph_results": graph_results,
            "api_results": api_results,
            "total_results": len(orphanet_results) + len(graph_results) + len(api_results)
        }
    except Exception as e:
        logger.error(f"Error searching diseases: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/drugs")
async def search_drugs(request: DrugSearchRequest):
    """Search for drug information using FDA API."""
    try:
        # Search FDA drug labels
        drug_labels = fda_collector.search_drug_labels(
            f"openfda.brand_name:{request.drug_name}",
            limit=request.max_results
        )
        
        # Get adverse events
        adverse_events = fda_collector.get_adverse_events(
            f"patient.drug.medicinalproduct:{request.drug_name}",
            limit=request.max_results
        )
        
        return {
            "drug_labels": drug_labels.to_dict('records'),
            "adverse_events": adverse_events.to_dict('records'),
            "total_results": len(drug_labels) + len(adverse_events)
        }
    except Exception as e:
        logger.error(f"Error searching drugs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-graph/stats")
async def get_knowledge_graph_stats():
    """Get statistics about the knowledge graph."""
    try:
        nodes = list(knowledge_graph.graph.nodes())
        edges = list(knowledge_graph.graph.edges())
        
        # Count by type
        node_types = {}
        for node in nodes:
            node_type = knowledge_graph.graph.nodes[node].get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": node_types,
            "diseases": len(knowledge_graph.disease_nodes),
            "articles": len(knowledge_graph.article_nodes)
        }
    except Exception as e:
        logger.error(f"Error getting knowledge graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-graph/add-disease")
async def add_disease_to_graph(disease_id: str):
    """Add a disease to the knowledge graph from Orphanet."""
    try:
        if not orphanet_collector:
            raise HTTPException(status_code=400, detail="Orphanet API not configured")
        
        disease = orphanet_collector.from_api(disease_id)
        knowledge_graph.add_disease(disease)
        
        # Save the updated graph
        knowledge_graph.save_graph("knowledge_graph.json")
        
        return {
            "message": f"Disease {disease.name} added to knowledge graph",
            "disease_id": disease_id,
            "disease_name": disease.name
        }
    except Exception as e:
        logger.error(f"Error adding disease to graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-graph/add-articles")
async def add_articles_to_graph(query: str, max_results: int = 5):
    """Add articles to the knowledge graph from PubMed."""
    try:
        if not pubmed_collector:
            raise HTTPException(status_code=400, detail="PubMed API not configured")
        
        articles = pubmed_collector.search(query, max_results=max_results)
        
        added_articles = []
        for article in articles:
            knowledge_graph.add_article(article)
            added_articles.append({
                "pmid": article.pmid,
                "title": article.title,
                "journal": article.journal
            })
        
        # Save the updated graph
        knowledge_graph.save_graph("knowledge_graph.json")
        
        return {
            "message": f"Added {len(added_articles)} articles to knowledge graph",
            "articles": added_articles
        }
    except Exception as e:
        logger.error(f"Error adding articles to graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "knowledge_graph_nodes": len(knowledge_graph.graph.nodes()),
        "knowledge_graph_edges": len(knowledge_graph.graph.edges()),
        "pubmed_available": pubmed_collector is not None,
        "orphanet_available": orphanet_collector is not None,
        "fda_available": True
    }

@app.get("/knowledge-graph/abstract")
async def get_abstract_knowledge_graph():
    """Return an abstract view of the knowledge graph (nodes and edges)."""
    try:
        nodes = [
            {
                "id": node,
                "type": knowledge_graph.graph.nodes[node].get("type", "unknown"),
                "name": knowledge_graph.graph.nodes[node].get("name", node)
            }
            for node in knowledge_graph.graph.nodes()
        ]
        edges = [
            {
                "source": source,
                "target": target,
                "type": knowledge_graph.graph.edges[source, target].get("relationship", "related_to")
            }
            for source, target in knowledge_graph.graph.edges()
        ]
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        logger.error(f"Error getting abstract knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-graph/from-collectors")
async def build_graph_from_collectors(
    disease_name: str = Body(..., embed=True),
    max_results: int = Body(3, embed=True)
):
    """Build knowledge graph from all collector outputs (PubMed, Orphanet, FDA) using LLM for graph unification."""
    try:
        collector_data = {
            "pubmed": {"entities": [], "relationships": []},
            "orphanet": {"entities": [], "relationships": []},
            "fda": {"entities": [], "relationships": []}
        }
        pubmed_articles = 0
        orphanet_diseases = 0
        fda_drugs = 0

        # --- PubMed ---
        if pubmed_collector:
            try:
                articles = pubmed_collector.search(disease_name, max_results=max_results)
                pubmed_articles = len(articles)
                for article in articles:
                    if article.abstract:
                        result = llm_entity_extractor.extract_entities_with_relationships(article.abstract)
                        collector_data["pubmed"]["entities"].extend(result["entities"])
                        collector_data["pubmed"]["relationships"].extend(result["relationships"])
            except Exception as e:
                logger.warning(f"PubMed search failed: {e}")

        # --- Orphanet ---
        if orphanet_collector:
            try:
                diseases = orphanet_collector.search_by_name(disease_name)
                orphanet_diseases = len(diseases)
                for disease in diseases:
                    disease_entity = {"text": disease.name, "type": "disease", "confidence": 1.0, "metadata": disease.to_dict()}
                    collector_data["orphanet"]["entities"].append(disease_entity)
                    for pheno in disease.phenotypes:
                        pheno_name = pheno.get("name", "")
                        if pheno_name:
                            pheno_entity = {"text": pheno_name, "type": "phenotype", "confidence": 1.0, "metadata": pheno}
                            collector_data["orphanet"]["entities"].append(pheno_entity)
                            relationship = {
                                "source": disease.name,
                                "target": pheno_name,
                                "type": "has_phenotype",
                                "confidence": 1.0
                            }
                            collector_data["orphanet"]["relationships"].append(relationship)
            except Exception as e:
                logger.warning(f"Orphanet search failed: {e}")

        # --- FDA ---
        if fda_collector:
            try:
                drug_labels = fda_collector.search_drug_labels(f"indications_and_usage:{disease_name}", limit=max_results)
                if drug_labels.empty:
                    drug_labels = fda_collector.search_drug_labels(f"description:{disease_name}", limit=max_results)
                fda_drugs = len(drug_labels)
                for _, row in drug_labels.iterrows():
                    drug_name = row['openfda'].get('brand_name', [disease_name])[0] if row['openfda'].get('brand_name') else disease_name
                    drug_entity = {"text": drug_name, "type": "drug", "confidence": 1.0, "metadata": row.to_dict()}
                    collector_data["fda"]["entities"].append(drug_entity)
                    try:
                        adverse_events = fda_collector.get_adverse_events(f"patient.drug.medicinalproduct:{drug_name}", limit=2)
                        for _, ae_row in adverse_events.iterrows():
                            for reaction in ae_row.get('reactions', []):
                                reaction_name = reaction.get('reactionmeddrapt', 'adverse_event')
                                if reaction_name and reaction_name != 'adverse_event':
                                    ae_entity = {"text": reaction_name, "type": "adverse_event", "confidence": 1.0, "metadata": reaction}
                                    collector_data["fda"]["entities"].append(ae_entity)
                                    relationship = {
                                        "source": drug_name,
                                        "target": reaction_name,
                                        "type": "causes_adverse_event",
                                        "confidence": 1.0
                                    }
                                    collector_data["fda"]["relationships"].append(relationship)
                    except Exception as ae_error:
                        logger.warning(f"Failed to get adverse events for {drug_name}: {ae_error}")
            except Exception as e:
                logger.warning(f"FDA search failed: {e}")

        # --- LLM Graph Unification ---
        # Prepare prompt for LLM
        prompt = f"""
You are a medical knowledge graph builder.

Here are entities and relationships from different sources for the disease '{disease_name}':

PubMed entities: {collector_data['pubmed']['entities']}
PubMed relationships: {collector_data['pubmed']['relationships']}
Orphanet entities: {collector_data['orphanet']['entities']}
Orphanet relationships: {collector_data['orphanet']['relationships']}
FDA entities: {collector_data['fda']['entities']}
FDA relationships: {collector_data['fda']['relationships']}

Unify these into a single knowledge graph. Remove duplicates, link related entities across sources, and output JSON with:
{{
  "nodes": [
    {{"id": "unique_id", "type": "disease/drug/symptom/phenotype/adverse_event/...", "name": "entity name"}}
  ],
  "edges": [
    {{"source": "unique_id", "target": "unique_id", "type": "relationship_type"}}
  ]
}}
"""
        
        llm_graph = None
        try:
            llm_graph = llm_entity_extractor.extract_graph_from_prompt(prompt)
        except Exception as e:
            logger.error(f"LLM graph unification failed: {e}")
            raise HTTPException(status_code=500, detail=f"LLM graph unification failed: {e}")

        # --- Build the graph from LLM output ---
        if not llm_graph or not isinstance(llm_graph, dict):
            raise HTTPException(status_code=500, detail="LLM did not return a valid graph JSON.")
        
        # Clear the current graph
        knowledge_graph.graph.clear()
        knowledge_graph.disease_nodes.clear()
        knowledge_graph.article_nodes.clear()
        knowledge_graph.mesh_term_nodes.clear()
        knowledge_graph.chemical_nodes.clear()
        
        # Add nodes
        for node in llm_graph.get("nodes", []):
            node_id = node["id"]
            node_type = node.get("type", "unknown")
            node_name = node.get("name", node_id)
            knowledge_graph.graph.add_node(node_id, type=node_type, name=node_name)
            if node_type == "disease":
                knowledge_graph.disease_nodes[node_id] = node
            elif node_type == "article":
                knowledge_graph.article_nodes[node_id] = node
            elif node_type == "mesh_term":
                knowledge_graph.mesh_term_nodes[node_id] = node
            elif node_type == "chemical":
                knowledge_graph.chemical_nodes[node_id] = node
        # Add edges
        for edge in llm_graph.get("edges", []):
            source = edge["source"]
            target = edge["target"]
            rel_type = edge.get("type", "related_to")
            knowledge_graph.graph.add_edge(source, target, relationship=rel_type)
        knowledge_graph.save_graph("knowledge_graph.json")
        
        # Return the new graph
        nodes = [
            {"id": node, "type": knowledge_graph.graph.nodes[node].get("type", "unknown"), "name": knowledge_graph.graph.nodes[node].get("name", node)}
            for node in knowledge_graph.graph.nodes()
        ]
        edges = [
            {"source": source, "target": target, "type": knowledge_graph.graph.edges[source, target].get("relationship", "related_to")}
            for source, target in knowledge_graph.graph.edges()
        ]
        summary = {
            "pubmed_articles": pubmed_articles,
            "orphanet_diseases": orphanet_diseases,
            "fda_drugs": fda_drugs,
            "total_entities": len(nodes),
            "total_relationships": len(edges),
            "graph_nodes": len(nodes),
            "graph_edges": len(edges)
        }
        return {
            "nodes": nodes,
            "edges": edges,
            "collector_summary": summary,
            "collector_data": collector_data
        }
    except Exception as e:
        logger.error(f"Error building graph from collectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RAG-Ready Graph Builder Endpoints
@app.post("/rag-ready/build-graph")
async def build_rag_ready_graph(
    query: str = Body(..., embed=True),
    max_results: int = Body(5, embed=True)
):
    """Build a RAG-ready knowledge graph using the RAG-ready graph builder."""
    try:
        # Initialize RAG-ready graph builder
        rag_builder = RAGReadyGraphBuilder()
        
        # Build the RAG-ready graph
        rag_structure = rag_builder.build_rag_ready_graph(query, max_results)
        
        # Get metadata
        metadata = rag_structure["metadata"]
        
        return {
            "success": True,
            "message": "RAG-ready graph built successfully",
            "metadata": metadata,
            "total_chunks": metadata["total_chunks"],
            "chunks_by_source": metadata["chunks_by_source"],
            "files_created": [
                "rag_ready_graph.json",
                "rag_chunks.json", 
                "rag_entity_nodes.json",
                "rag_retrieval_examples.json"
            ]
        }
    except Exception as e:
        logger.error(f"Error building RAG-ready graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag-ready/query")
async def query_rag_ready_graph(
    query: str = Body(..., embed=True),
    top_k: int = Body(5, embed=True)
):
    """Query the RAG-ready graph for relevant chunks."""
    try:
        # Check if RAG-ready graph exists
        if not os.path.exists("rag_ready_graph.json"):
            raise HTTPException(
                status_code=404, 
                detail="RAG-ready graph not found. Please build it first using /rag-ready/build-graph"
            )
        
        # Load RAG-ready graph
        with open("rag_ready_graph.json", "r") as f:
            rag_structure = json.load(f)
        
        # Initialize RAG builder and rebuild index
        rag_builder = RAGReadyGraphBuilder()
        chunks = rag_structure.get("chunks", [])
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No chunks found in RAG-ready graph"
            )
        
        # Rebuild FAISS index
        rag_builder.build_faiss_index(chunks)
        
        # Retrieve relevant chunks
        relevant_chunks = rag_builder.retrieve_relevant_chunks(query, top_k)
        
        return {
            "success": True,
            "query": query,
            "total_chunks_retrieved": len(relevant_chunks),
            "relevant_chunks": [
                {
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "type": chunk["type"],
                    "source": chunk["source"],
                    "retrieval_score": chunk.get("retrieval_score", 0.0),
                    "metadata": chunk.get("metadata", {})
                }
                for chunk in relevant_chunks
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying RAG-ready graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag-ready/simple-query")
async def simple_rag_query(
    query: str = Body(..., embed=True),
    disease: str = Body("cholera", embed=True),
    max_results: int = Body(3, embed=True)
):
    """Simple RAG query that builds graph on-the-fly and answers questions."""
    try:
        # Initialize RAG builder
        rag_builder = RAGReadyGraphBuilder()
        
        # Build RAG-ready graph for the disease
        logger.info(f"Building RAG-ready graph for {disease}")
        rag_structure = rag_builder.build_rag_ready_graph(disease, max_results)
        
        # Initialize RAG engine
        kg = KnowledgeGraph()
        rag_engine = RAGEngine(kg)
        
        # Get answer using RAG-ready method
        result = rag_engine.answer_question(query, use_rag_ready=True)
        
        return {
            "success": True,
            "query": query,
            "disease": disease,
            "answer": result["answer"],
            "confidence": result.get("confidence", 0.0),
            "follow_up_question": result.get("follow_up_question"),
            "context_sources": len(result.get("context", [])),
            "method": "rag_ready_on_the_fly"
        }
        
    except Exception as e:
        logger.error(f"Error in simple RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port) 