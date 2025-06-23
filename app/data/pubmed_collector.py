import requests
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import time
from urllib.parse import quote
import os
import json
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

class PubMedArticle:
    """Class to represent a PubMed article with comprehensive metadata."""

    def __init__(
        self,
        pmid: str,
        title: str,
        abstract: Optional[str] = None,
        authors: Optional[List[str]] = None,
        journal: Optional[str] = None,
        publication_date: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        mesh_terms: Optional[List[Dict[str, Any]]] = None,
        chemicals: Optional[List[Dict[str, Any]]] = None,
        grants: Optional[List[Dict[str, Any]]] = None,
        references: Optional[List[Dict[str, Any]]] = None
    ):
        self.pmid = pmid
        self.title = title
        self.abstract = abstract
        self.authors = authors or []
        self.journal = journal
        self.publication_date = publication_date
        self.keywords = keywords or []
        self.metadata = metadata or {}
        self.mesh_terms = mesh_terms or []
        self.chemicals = chemicals or []
        self.grants = grants or []
        self.references = references or []

    @classmethod
    def from_api(cls, pmid: str, api_key: Optional[str] = None) -> "PubMedArticle":
        """Fetch article data from PubMed API and return a PubMedArticle instance.
        
        Args:
            pmid: PubMed ID of the article
            api_key: Optional API key for PubMed. If not provided, will look for PUBMED_API_KEY env var.
            
        Returns:
            PubMedArticle instance with data from the API
            
        Raises:
            ValueError: If no article data is found for the given PMID
        """
        api_key = api_key or os.getenv("PUBMED_API_KEY")
        if not api_key:
            raise ValueError("PubMed API key is required. Set PUBMED_API_KEY environment variable or pass api_key parameter.")
        
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Fetch article summary
        summary_url = f"{base_url}/esummary.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "json",
            "api_key": api_key
        }
        
        try:
            response = requests.get(summary_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            logging.info(f"PubMed API Response for PMID {pmid}: {json.dumps(data, indent=2)}")
            
            # Extract article data from the response
            if "result" not in data or pmid not in data["result"]:
                raise ValueError(f"No article data found for PMID {pmid}")
            
            article_data = data["result"][pmid]
            logging.info(f"Article data extracted: {json.dumps(article_data, indent=2)}")
            
            # Extract authors
            authors = []
            if "authors" in article_data:
                for author in article_data["authors"]:
                    if isinstance(author, dict) and "name" in author:
                        authors.append(author["name"])
            
            # Extract MeSH terms
            mesh_terms = []
            if "meshterms" in article_data:
                for term in article_data["meshterms"]:
                    if isinstance(term, dict) and "name" in term:
                        mesh_terms.append(term["name"])
            
            # Extract chemicals
            chemicals = []
            if "chemicals" in article_data:
                for chem in article_data["chemicals"]:
                    if isinstance(chem, dict) and "name" in chem:
                        chemicals.append(chem["name"])
            
            # Extract grants
            grants = []
            if "grants" in article_data:
                for grant in article_data["grants"]:
                    if isinstance(grant, dict) and "agency" in grant:
                        grants.append(grant["agency"])
            
            # Extract references
            references = []
            if "references" in article_data:
                for ref in article_data["references"]:
                    if isinstance(ref, dict) and "pmid" in ref:
                        references.append(ref["pmid"])
            
            # Fetch abstract using efetch
            efetch_url = f"{base_url}/efetch.fcgi"
            efetch_params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "xml",
                "api_key": api_key
            }
            efetch_response = requests.get(efetch_url, params=efetch_params)
            efetch_response.raise_for_status()
            root = ET.fromstring(efetch_response.text)
            abstract_text = ""
            for abstract in root.findall(".//AbstractText"):
                if abstract.text:
                    abstract_text += abstract.text + " "
            abstract_text = abstract_text.strip()
            
            # Create metadata dictionary
            metadata = {
                "source": "PubMed",
                "last_updated": datetime.now().isoformat(),
                "raw_data": article_data
            }
            
            return cls(
                pmid=pmid,
                title=article_data.get("title", ""),
                authors=authors,
                abstract=abstract_text or article_data.get("abstract", ""),
                journal=article_data.get("fulljournalname", ""),
                publication_date=article_data.get("pubdate", ""),
                mesh_terms=mesh_terms,
                chemicals=chemicals,
                grants=grants,
                references=references,
                metadata=metadata
            )
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch article {pmid}: {str(e)}")
            raise ValueError(f"Failed to fetch article {pmid}: {str(e)}")

    @classmethod
    def search(
        cls,
        query: str,
        max_results: int = 10,
        api_key: Optional[str] = None,
        sort: str = "relevance",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> List['PubMedArticle']:
        """
        Search PubMed articles based on query string.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            api_key: Optional API key for PubMed API access
            sort: Sort order ('relevance', 'date', 'pub_date')
            min_date: Minimum publication date (YYYY/MM/DD)
            max_date: Maximum publication date (YYYY/MM/DD)
            
        Returns:
            List of PubMedArticle instances
        """
        if not api_key:
            api_key = os.getenv('PUBMED_API_KEY')

        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Build search query
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": sort,
            "api_key": api_key
        }
        
        if min_date:
            search_params["mindate"] = min_date
        if max_date:
            search_params["maxdate"] = max_date

        try:
            # First, get the list of PMIDs
            response = requests.get(f"{base_url}/esearch.fcgi", params=search_params)
            response.raise_for_status()
            search_data = response.json()
            
            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            articles = []
            
            # Then fetch details for each PMID
            for pmid in pmids:
                try:
                    article = cls.from_api(pmid, api_key)
                    articles.append(article)
                    # Respect rate limiting
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Failed to fetch article {pmid}: {str(e)}")
                    continue
            
            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search PubMed articles: {str(e)}")
            raise

    @staticmethod
    def _fetch_mesh_terms(pmid: str, api_key: str) -> List[Dict[str, Any]]:
        """Fetch MeSH terms for an article."""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "json",
            "api_key": api_key
        }

        try:
            response = requests.get(f"{base_url}/efetch.fcgi", params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("PubmedArticle", [{}])[0].get("MedlineCitation", {}).get("MeshHeadingList", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch MeSH terms for article {pmid}: {str(e)}")
            return []

    @staticmethod
    def _fetch_chemicals(pmid: str, api_key: str) -> List[Dict[str, Any]]:
        """Fetch chemical substances for an article."""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "json",
            "api_key": api_key
        }

        try:
            response = requests.get(f"{base_url}/efetch.fcgi", params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("PubmedArticle", [{}])[0].get("MedlineCitation", {}).get("ChemicalList", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch chemicals for article {pmid}: {str(e)}")
            return []

    @staticmethod
    def _fetch_grants(pmid: str, api_key: str) -> List[Dict[str, Any]]:
        """Fetch grant information for an article."""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "json",
            "api_key": api_key
        }

        try:
            response = requests.get(f"{base_url}/efetch.fcgi", params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("PubmedArticle", [{}])[0].get("MedlineCitation", {}).get("Article", {}).get("GrantList", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch grants for article {pmid}: {str(e)}")
            return []

    @staticmethod
    def _fetch_references(pmid: str, api_key: str) -> List[Dict[str, Any]]:
        """Fetch references for an article."""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "json",
            "api_key": api_key
        }

        try:
            response = requests.get(f"{base_url}/efetch.fcgi", params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("PubmedArticle", [{}])[0].get("PubmedData", {}).get("ReferenceList", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch references for article {pmid}: {str(e)}")
            return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert the article instance to a dictionary representation."""
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "publication_date": self.publication_date,
            "keywords": self.keywords,
            "metadata": self.metadata,
            "mesh_terms": self.mesh_terms,
            "chemicals": self.chemicals,
            "grants": self.grants,
            "references": self.references
        }

    def __str__(self) -> str:
        """String representation of the article."""
        return f"PubMedArticle(pmid={self.pmid}, title={self.title})"

    def __repr__(self) -> str:
        """Detailed string representation of the article."""
        return (
            f"PubMedArticle("
            f"pmid='{self.pmid}', "
            f"title='{self.title}', "
            f"journal='{self.journal}', "
            f"publication_date='{self.publication_date}', "
            f"authors={self.authors}, "
            f"keywords={self.keywords}"
            f")"
        ) 