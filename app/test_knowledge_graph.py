import os
import logging
from app.core.knowledge_graph import KnowledgeGraph
from app.data.orphanet_collector import OrphanetDisease
from app.data.pubmed_collector import PubMedArticle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_knowledge_graph():
    """Test the knowledge graph functionality with Orphanet and PubMed data."""
    # Initialize knowledge graph
    kg = KnowledgeGraph()
    
    # Test 1: Add a disease from Orphanet
    try:
        # Fetch Marfan syndrome (ORPHA:558)
        disease = OrphanetDisease.from_api("558")
        kg.add_disease(disease)
        logger.info(f"Added disease: {disease.name}")
        
        # Search for related articles
        articles = PubMedArticle.search(f"{disease.name} syndrome", max_results=5)
        logger.info(f"Found {len(articles)} related articles")
        
        # Add articles to the graph
        for article in articles:
            kg.add_article(article)
            kg.link_disease_to_article(disease.disease_id, article.pmid)
            logger.info(f"Added and linked article: {article.title}")
        
        # Test 2: Search functionality
        disease_results = kg.search_diseases("Marfan")
        logger.info(f"Found {len(disease_results)} diseases matching 'Marfan'")
        
        article_results = kg.search_articles("Marfan syndrome")
        logger.info(f"Found {len(article_results)} articles matching 'Marfan syndrome'")
        
        # Test 3: Network analysis
        disease_network = kg.get_disease_network(disease.disease_id)
        logger.info(f"Disease network contains {disease_network.number_of_nodes()} nodes and {disease_network.number_of_edges()} edges")
        
        # Test 4: Get related articles
        related_articles = kg.get_disease_articles(disease.disease_id)
        logger.info(f"Found {len(related_articles)} articles related to {disease.name}")
        
        # Test 5: Save and load graph
        graph_file = "test_graph.pkl"
        kg.save_graph(graph_file)
        print(f"Knowledge graph saved to: {os.path.abspath(graph_file)}")
        print(f"Number of nodes: {kg.graph.number_of_nodes()}")
        print(f"Number of edges: {kg.graph.number_of_edges()}")
        loaded_kg = KnowledgeGraph.load_graph("test_graph.pkl")
        logger.info("Successfully saved and loaded knowledge graph")
        
        # Clean up
        os.remove("test_graph.pkl")
        
    except Exception as e:
        logger.error(f"Error during knowledge graph testing: {str(e)}")
        raise

def main():
    """Main function to run the knowledge graph tests."""
    # Check for required API keys
    if not os.getenv('ORPHANET_API_KEY'):
        logger.error("ORPHANET_API_KEY not set")
        return
    
    if not os.getenv('PUBMED_API_KEY'):
        logger.error("PUBMED_API_KEY not set")
        return
    
    # Run tests
    test_knowledge_graph()

if __name__ == "__main__":
    main() 