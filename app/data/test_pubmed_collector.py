import unittest
import os
from app.data.pubmed_collector import PubMedArticle

class TestPubMedCollector(unittest.TestCase):
    def setUp(self):
        # Check if API key is available
        self.api_key = os.getenv('PUBMED_API_KEY')
        if not self.api_key:
            self.skipTest("PUBMED_API_KEY environment variable not set")

    def test_pubmed_article_creation(self):
        """Test creating a PubMedArticle instance with basic data."""
        article = PubMedArticle(
            pmid="12345",
            title="Test Article",
            abstract="This is a test abstract",
            authors=["Author 1", "Author 2"]
        )
        
        self.assertEqual(article.pmid, "12345")
        self.assertEqual(article.title, "Test Article")
        self.assertEqual(article.abstract, "This is a test abstract")
        self.assertEqual(len(article.authors), 2)

    def test_pubmed_search(self):
        """Test PubMed search functionality."""
        if not self.api_key:
            self.skipTest("PUBMED_API_KEY not available")
        
        # Test search with a simple query
        articles = PubMedArticle.search(
            query="cancer",
            max_results=2,
            api_key=self.api_key
        )
        
        self.assertIsInstance(articles, list)
        self.assertLessEqual(len(articles), 2)
        
        if articles:
            article = articles[0]
            self.assertIsInstance(article, PubMedArticle)
            self.assertIsNotNone(article.pmid)
            self.assertIsNotNone(article.title)

    def test_pubmed_article_from_api(self):
        """Test fetching a specific article by PMID."""
        if not self.api_key:
            self.skipTest("PUBMED_API_KEY not available")
        
        # Use a known PMID for testing
        try:
            article = PubMedArticle.from_api("1234567", self.api_key)
            self.assertIsInstance(article, PubMedArticle)
            self.assertEqual(article.pmid, "1234567")
        except ValueError:
            # This is expected if the PMID doesn't exist
            pass

    def test_pubmed_article_to_dict(self):
        """Test converting PubMedArticle to dictionary."""
        article = PubMedArticle(
            pmid="12345",
            title="Test Article",
            abstract="Test abstract",
            authors=["Author 1"]
        )
        
        article_dict = article.to_dict()
        self.assertIsInstance(article_dict, dict)
        self.assertEqual(article_dict["pmid"], "12345")
        self.assertEqual(article_dict["title"], "Test Article")

if __name__ == "__main__":
    unittest.main() 