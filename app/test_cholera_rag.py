import logging
import argparse
from app.core.knowledge_graph import KnowledgeGraph
from app.rag.rag_engine import RAGEngine

def parse_args():
    parser = argparse.ArgumentParser(description='Test cholera RAG system')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load the knowledge graph (Cholera-specific)
    logger.info("Loading knowledge graph for Cholera...")
    knowledge_graph = KnowledgeGraph()
    knowledge_graph.load_from_file("data/knowledge_graph_20250518_162509.json")
    
    # Initialize RAG engine
    logger.info("Initializing RAG engine...")
    rag_engine = RAGEngine(knowledge_graph)
    
    # Cholera-specific test questions
    test_questions = [
        "What are the symptoms of cholera?",
        "How is cholera transmitted?",
        "What are the recommended treatments for cholera?",
        "What are the complications of untreated cholera?"
    ]
    
    # Get answers
    for question in test_questions:
        logger.info(f"\nQuestion: {question}")
        result = rag_engine.answer_question(question)
        
        logger.info("Answer:")
        logger.info(result['answer'])
        
        logger.info("\nContext used:")
        for item in result['context']:
            logger.info(f"- {item['type']}: {item['name']}")
            if item['metadata']:
                logger.info("  Metadata:")
                for key, value in item['metadata'].items():
                    logger.info(f"    {key}: {value}")
        
        logger.info("-" * 50)

if __name__ == "__main__":
    main() 