import os
import logging
from app.data.orphanet_collector import OrphanetDisease
from app.data.pubmed_collector import PubMedArticle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_orphanet_collector():
    """Test the OrphanetDisease collector functionality."""
    logger.info("Testing OrphanetDisease collector...")
    
    # Test 1: Fetch Marfan syndrome (ORPHA:558)
    try:
        disease = OrphanetDisease.from_api("558")
        logger.info(f"Successfully fetched disease: {disease}")
        logger.info(f"Name: {disease.name}")
        logger.info(f"Prevalence: {disease.prevalence}")
        logger.info(f"Medical specialties: {disease.medical_specialties}")
    except Exception as e:
        logger.error(f"Failed to fetch Orphanet disease: {str(e)}")

def test_pubmed_collector():
    """Test the PubMedArticle collector functionality."""
    logger.info("Testing PubMedArticle collector...")
    
    # Test 1: Search for cholera articles
    try:
        articles = PubMedArticle.search("cholera treatment", max_results=3)
        logger.info(f"Found {len(articles)} articles")
        
        for article in articles:
            logger.info(f"\nArticle: {article.title}")
            logger.info(f"Authors: {', '.join(article.authors)}")
            logger.info(f"Journal: {article.journal}")
            logger.info(f"Publication date: {article.publication_date}")
            if article.abstract:
                logger.info(f"Abstract preview: {article.abstract[:200]}...")
    except Exception as e:
        logger.error(f"Failed to search PubMed articles: {str(e)}")
    
    # Test 2: Fetch specific article
    try:
        article = PubMedArticle.from_api("12345678")  # Replace with a valid PMID
        logger.info(f"\nFetched specific article: {article.title}")
        logger.info(f"Authors: {', '.join(article.authors)}")
        logger.info(f"Journal: {article.journal}")
    except Exception as e:
        logger.error(f"Failed to fetch specific PubMed article: {str(e)}")

def main():
    """Main function to run all tests."""
    # Check for required API keys
    if not os.getenv('ORPHANET_API_KEY'):
        logger.error("ORPHANET_API_KEY not set")
        return
    
    if not os.getenv('PUBMED_API_KEY'):
        logger.error("PUBMED_API_KEY not set")
        return
    
    # Run tests
    test_orphanet_collector()
    test_pubmed_collector()

if __name__ == "__main__":
    main() 