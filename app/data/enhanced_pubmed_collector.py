#!/usr/bin/env python3
"""
Enhanced PubMed Collector with PMC Full-Text Support

This module provides an enhanced PubMed collector that can:
- Search PubMed articles
- Fetch comprehensive metadata
- Extract full-text from PMC when available
- Output in knowledge graph node format
- Handle batching and rate limiting

Usage:
    export NCBI_API_KEY="<your-api-key>"
    export NCBI_EMAIL="<your@email>"
    
    # As a module
    from app.data.enhanced_pubmed_collector import EnhancedPubMedCollector
    collector = EnhancedPubMedCollector()
    nodes = collector.search_and_fetch("diabetes", max_results=50)
    
    # As a script
    python -m app.data.enhanced_pubmed_collector -q "diabetes" -n 50
"""

import os
import json
import argparse
import requests
import xml.etree.ElementTree as ET
from time import sleep
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EnhancedPubMedCollector:
    """Enhanced PubMed collector with PMC full-text support."""
    
    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        """
        Initialize the collector.
        
        Args:
            api_key: NCBI API key (will use NCBI_API_KEY env var if not provided)
            email: Contact email (will use NCBI_EMAIL env var if not provided)
        """
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        self.email = email or os.getenv("NCBI_EMAIL")
        
        if not (self.api_key and self.email):
            raise ValueError("NCBI_API_KEY and NCBI_EMAIL must be provided via env vars or parameters")
        
        self.base_search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.base_fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def search_pmids(self, query: str, limit: int) -> List[str]:
        """Search PubMed and return PMIDs."""
        params = {
            "db": "pubmed",
            "term": f"{query}[TIAB]",
            "retmax": limit,
            "retmode": "xml",
            "api_key": self.api_key,
            "email": self.email,
        }
        
        try:
            response = requests.get(self.base_search_url, params=params, timeout=30)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            return [e.text for e in root.findall(".//Id")]
        except Exception as e:
            logger.error(f"Failed to search PubMed: {e}")
            return []
    
    def batch_fetch_xml(self, pmids: List[str], batch_size: int = 20) -> List[ET.Element]:
        """Fetch PubMed articles in batches."""
        roots = []
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i+batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "api_key": self.api_key,
                "email": self.email,
            }
            
            try:
                response = requests.get(self.base_fetch_url, params=params, timeout=60)
                response.raise_for_status()
                roots.append(ET.fromstring(response.text))
                sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to fetch batch {batch}: {e}")
        
        return roots
    
    def parse_pubmed_article(self, art: ET.Element) -> Dict[str, Any]:
        """Parse a single PubMed article from XML."""
        def get_text(path: str, default: str = "") -> str:
            return art.findtext(path, default=default)
        
        pmid = get_text(".//PMID")
        title = get_text(".//ArticleTitle")
        
        # Extract abstract
        abstract_parts = []
        for abstract_text in art.findall(".//Abstract/AbstractText"):
            text = ET.tostring(abstract_text, method="text", encoding="unicode").strip()
            if text:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)
        
        # Extract authors
        authors = []
        for author in art.findall(".//AuthorList/Author"):
            last_name = author.findtext("LastName")
            first_name = author.findtext("ForeName") or author.findtext("Initials")
            if last_name and first_name:
                authors.append(f"{first_name} {last_name}")
            elif last_name:
                authors.append(last_name)
        
        journal = get_text(".//Journal/Title")
        
        # Extract publication date
        pub_el = art.find(".//Journal/JournalIssue/PubDate")
        if pub_el is not None:
            year = pub_el.findtext("Year", "")
            month = pub_el.findtext("Month", "")
            day = pub_el.findtext("Day", "")
            if year and month and day:
                pubdate = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            else:
                pubdate = pub_el.findtext("MedlineDate") or year
        else:
            pubdate = ""
        
        # Extract keywords
        keywords = [kw.text for kw in art.findall(".//Keyword") if kw.text]
        
        # Extract MeSH terms
        mesh_terms = []
        for mesh in art.findall(".//MeshHeadingList/MeshHeading"):
            descriptor = mesh.find("DescriptorName")
            if descriptor is not None and descriptor.text:
                mesh_terms.append(descriptor.text)
        
        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "publication_date": pubdate,
            "keywords": keywords,
            "mesh_terms": mesh_terms,
        }
    
    def _strip_tag(self, tag: str) -> str:
        """Strip namespace from XML tag."""
        return tag.split('}', 1)[-1] if '}' in tag else tag
    
    def _extract_text(self, element) -> str:
        """Extract text from XML element."""
        return "".join(element.itertext()).strip() if element is not None else ""
    
    def _flatten_body(self, body: ET.Element) -> List[str]:
        """Flatten PMC article body into text sections."""
        out = []
        for child in body:
            name = self._strip_tag(child.tag)
            if name in ("title", "sec"):
                title_text = self._extract_text(child.find("./title"))
                if title_text:
                    out.append(f"\n\n## {title_text}\n")
                out.extend(self._flatten_body(child))
            elif name == "p":
                paragraph = self._extract_text(child)
                if paragraph:
                    out.append(paragraph)
        return out
    
    def fetch_fulltext_safe(self, pmid: str) -> str:
        """Safely fetch full-text from PMC if available."""
        try:
            # Convert PMID to PMCID
            conv_response = requests.get(
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                params={
                    "ids": pmid, 
                    "format": "json", 
                    "api_key": self.api_key, 
                    "email": self.email
                },
                timeout=20,
            )
            conv_response.raise_for_status()
            conv_data = conv_response.json()
            
            if "records" not in conv_data or not conv_data["records"]:
                return ""
            
            pmcid = conv_data["records"][0].get("pmcid")
            if not pmcid:
                return ""
            
            # Fetch PMC full-text
            pmc_params = {
                "db": "pmc", 
                "id": pmcid, 
                "rettype": "xml", 
                "retmode": "xml",
                "api_key": self.api_key, 
                "email": self.email
            }
            
            pmc_response = requests.get(self.base_fetch_url, params=pmc_params, timeout=60)
            pmc_response.raise_for_status()
            
            root = ET.fromstring(pmc_response.text)
            
            # Strip namespaces
            for element in root.iter():
                element.tag = self._strip_tag(element.tag)
            
            # Extract content
            title = self._extract_text(root.find(".//article-title"))
            abstract = " ".join(
                self._extract_text(p) for p in root.findall(".//abstract//p")
            ).strip()
            
            body = []
            body_element = root.find(".//body")
            if body_element is not None:
                body = self._flatten_body(body_element)
            
            sleep(0.2)  # Rate limiting
            
            return f"# {title}\n\n### Abstract\n{abstract}\n\n### Full Text\n" + "\n".join(body)
            
        except Exception as e:
            logger.warning(f"Failed to fetch full-text for PMID {pmid}: {e}")
            return ""
    
    def build_knowledge_graph_nodes(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert articles to knowledge graph node format."""
        nodes = []
        
        for article in articles:
            # Fetch full-text if available
            fulltext = self.fetch_fulltext_safe(article["pmid"])
            
            node = {
                "type": "article",
                "name": article["title"],
                "id": f"pubmed_{article['pmid']}",
                "metadata": {
                    "pmid": article["pmid"],
                    "abstract": article["abstract"],
                    "authors": article["authors"],
                    "journal": article["journal"],
                    "publication_date": article["publication_date"],
                    "keywords": article["keywords"],
                    "mesh_terms": article.get("mesh_terms", []),
                    "fulltext": fulltext,
                    "source": "PubMed",
                    "collected_at": datetime.now().isoformat(),
                }
            }
            
            nodes.append(node)
        
        return {"nodes": nodes}
    
    def search_and_fetch(
        self, 
        query: str, 
        max_results: int = 50, 
        batch_size: int = 20,
        include_fulltext: bool = True
    ) -> Dict[str, Any]:
        """
        Search PubMed and fetch articles with optional full-text.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            batch_size: Batch size for API calls
            include_fulltext: Whether to fetch full-text from PMC
            
        Returns:
            Dictionary with knowledge graph nodes
        """
        logger.info(f"Searching PubMed for: {query}")
        
        # Search for PMIDs
        pmids = self.search_pmids(query, max_results)
        logger.info(f"Found {len(pmids)} PMIDs")
        
        if not pmids:
            return {"nodes": []}
        
        # Fetch article data
        roots = self.batch_fetch_xml(pmids, batch_size)
        articles = []
        for root in roots:
            for article_elem in root.findall(".//PubmedArticle"):
                article = self.parse_pubmed_article(article_elem)
                articles.append(article)
        
        logger.info(f"Parsed {len(articles)} articles")
        
        # Build knowledge graph nodes
        if include_fulltext:
            logger.info("Fetching full-text from PMC...")
        
        nodes = self.build_knowledge_graph_nodes(articles)
        
        return nodes
    
    def save_to_file(self, data: Dict[str, Any], filename: str) -> None:
        """Save data to JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved to {filename}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Enhanced PubMed → JSON nodes (+ optional PMC full-text)")
    parser.add_argument("-q", "--query", required=True, help="Search term (TIAB)")
    parser.add_argument("-n", "--max", type=int, default=50, help="Max PubMed hits")
    parser.add_argument("-b", "--batch", type=int, default=20, help="EFetch batch size")
    parser.add_argument("-o", "--output", help="Output JSON filename")
    parser.add_argument("--api_key", help="NCBI API key (overrides $NCBI_API_KEY)")
    parser.add_argument("--email", help="Contact e-mail (overrides $NCBI_EMAIL)")
    parser.add_argument("--no-fulltext", action="store_true", help="Skip full-text extraction")
    
    args = parser.parse_args()
    
    try:
        collector = EnhancedPubMedCollector(
            api_key=args.api_key,
            email=args.email
        )
        
        nodes = collector.search_and_fetch(
            query=args.query,
            max_results=args.max,
            batch_size=args.batch,
            include_fulltext=not args.no_fulltext
        )
        
        outfile = args.output or f"{args.query.lower().replace(' ', '_')}_nodes.json"
        collector.save_to_file(nodes, outfile)
        
        print(f"✓ Successfully processed {len(nodes['nodes'])} articles")
        print(f"✓ Saved to {outfile}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
