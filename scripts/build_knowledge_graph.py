import os
import json
import logging
import argparse
from datetime import datetime, timedelta
from app.data.orphanet_collector import OrphanetCollector
from app.data.pubmed_collector import PubMedCollector
from app.data.pipeline import DataPipeline
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Build medical knowledge graph')
    parser.add_argument('--icd11-code', type=str, required=True, help='ICD-11 code to search for')
    parser.add_argument('--query', type=str, required=True, help='Search query for PubMed')
    parser.add_argument('--max-diseases', type=int, default=5, help='Maximum number of diseases to collect')
    parser.add_argument('--max-articles', type=int, default=10, help='Maximum number of articles per disease')
    parser.add_argument('--days-back', type=int, default=365, help='Number of days to look back for articles')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory to store data')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh of cached data')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Initialize collectors
    orphanet_collector = OrphanetCollector(api_key=settings.orphanet_api_key)
    pubmed_collector = PubMedCollector(
        api_key=settings.pubmed_api_key,
        email=settings.pubmed_email,
        tool_name=settings.pubmed_tool_name
    )
    
    # Initialize pipeline
    pipeline = DataPipeline(
        orphanet_collector=orphanet_collector,
        pubmed_collector=pubmed_collector,
        data_dir=args.data_dir
    )
    
    # Build knowledge graph
    logger.info(f"Building knowledge graph for ICD-11 code: {args.icd11_code}")
    logger.info(f"Search query: {args.query}")
    logger.info(f"Max diseases: {args.max_diseases}")
    logger.info(f"Max articles per disease: {args.max_articles}")
    logger.info(f"Days back: {args.days_back}")
    
    # Collect diseases
    diseases = pipeline.collect_rare_diseases(
        icd11_code=args.icd11_code,
        max_diseases=args.max_diseases,
        force_refresh=args.force_refresh
    )
    
    if not diseases:
        logger.error(f"No diseases found for ICD-11 code: {args.icd11_code}")
        return
    
    logger.info(f"Found {len(diseases)} diseases")
    
    # Collect literature for each disease
# Collect literature for each disease
    all_articles = {}

    for disease in diseases:
        logger.info(f"Collecting literature for disease: {disease['name']}")
        articles = pipeline.collect_medical_literature(
            diseases=[disease],
            max_articles_per_disease=args.max_articles,
            days_back=args.days_back,
            force_refresh=args.force_refresh
        )
        if articles:
            all_articles.update(articles)  # ✅ merge into final dict
            logger.info(f"Collected {sum(len(v) for v in articles.values())} articles for {disease['name']}")

    # Build knowledge graph
    logger.info("Building knowledge graph...")
    print("Type of all_articles:", type(all_articles))  # Should be <class 'dict'>
    graph = pipeline.build_knowledge_graph(diseases, all_articles)

    
    # Save knowledge graph
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_file = os.path.join(args.data_dir, f"knowledge_graph_{timestamp}.json")
    
    from networkx.readwrite import json_graph

    with open(graph_file, 'w') as f:
        json.dump(json_graph.node_link_data(graph.graph), f, indent=2)
    
    logger.info(f"Knowledge graph saved to: {graph_file}")
    logger.info(f"Graph contains {len(graph.graph.nodes)} nodes and {len(graph.graph.edges)} edges")


if __name__ == "__main__":
    main() 